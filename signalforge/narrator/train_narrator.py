"""
signalforge/narrator/train_narrator.py
LoRA fine-tunes a small local base model on the (prompt -> narration) dataset
from build_dataset.py, merges the adapter, converts to GGUF, and registers a
brand-new versioned tag in Ollama — never overwriting a previously registered
narrator version, same immutable-package philosophy as train/registry.py.

CPU-only on this machine (no discrete GPU) — this is why the base model is
tiny (Qwen2.5-0.5B-Instruct by default) and why this is a batch job you kick
off and wait on, not something run per-request. Expect tens of minutes for a
small dataset even at this size.

Registration goes through Ollama's HTTP API (POST /api/create) rather than
shelling out to the `ollama` CLI — the CLI isn't reliably on PATH in every
shell/user context, but the API is already how this app talks to Ollama
elsewhere (see localai/engine.py), and it's confirmed reachable at
localhost:11434 regardless of CLI visibility.
"""
import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

_THIS_DIR = Path(__file__).resolve().parent

log = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OLLAMA_BASE = "http://localhost:11434"
TAG_PREFIX = "signalforge-narrator"

MERGED_DIR = _THIS_DIR / "merged"
GGUF_DIR = _THIS_DIR / "gguf"
LLAMA_CPP_DIR = _THIS_DIR / ".llama_cpp"
NARRATOR_REGISTRY = _THIS_DIR / "registry.json"

SYSTEM_PROMPT = """You are the SignalForge narrator, running locally on this machine. You are given a stock \
ticker, a signal (BUY/SELL/HOLD) and confidence already decided by a separate trained model, backtested \
accuracy for that confidence level, the model's engineered technical features, and a live qualitative \
context blurb. Write 2-4 fluent, confident, analyst-style sentences weaving these together. Never invent a \
number that wasn't given to you, never change the signal or confidence, never add hedging beyond what the \
inputs already imply."""


# ── Dataset loading ────────────────────────────────────────────────────────

def _load_dataset(path: Path) -> list[dict]:
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


# ── LoRA fine-tuning ───────────────────────────────────────────────────────

def _finetune(examples: list[dict], output_dir: Path, epochs: int, base_model: str) -> Path:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq

    log.info("Loading base model %s (CPU)...", base_model)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def _tokenize(example: dict) -> dict:
        prompt_text = (
            f"Structured inputs:\n{example['prompt']}\n\nWrite the narrative."
        )
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        # Render the template as text and tokenize it ourselves rather than
        # relying on apply_chat_template(tokenize=True)'s return shape, which
        # varies across transformers versions (plain list vs. BatchEncoding).
        prompt_text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(example["completion"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        max_len = 1024   # narratives are a few sentences — this comfortably covers prompt + completion
        input_ids = (prompt_ids + completion_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + completion_ids)[:max_len]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    ds = Dataset.from_list(examples).map(_tokenize, remove_columns=["prompt", "completion"])

    checkpoints_dir = output_dir / "checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        # A ~5-hour CPU run on a personal machine (sleep/power events, a stray
        # process, anything) can die with nothing to show for it otherwise —
        # save_strategy="no" was exactly that mistake on the first full-scale
        # run. Checkpointing regularly makes a future interruption resumable
        # instead of a total loss.
        save_strategy="steps",
        save_steps=20,
        save_total_limit=2,
        report_to=[],
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    existing_checkpoints = sorted(checkpoints_dir.glob("checkpoint-*")) if checkpoints_dir.exists() else []
    resume_from = str(existing_checkpoints[-1]) if existing_checkpoints else None
    if resume_from:
        log.info("Resuming from checkpoint: %s", resume_from)

    log.info("Starting LoRA fine-tuning: %d examples, %d epochs (CPU — this will take a while)...", len(examples), epochs)
    trainer.train(resume_from_checkpoint=resume_from)

    log.info("Merging LoRA adapter into base weights...")
    merged = model.merge_and_unload()
    merged_path = output_dir / "merged"
    merged_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    return merged_path


# ── GGUF conversion (via llama.cpp's conversion script) ───────────────────

def _ensure_llama_cpp() -> Path:
    script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if script.exists():
        return script
    log.info("llama.cpp conversion script not found — cloning ggerganov/llama.cpp (shallow, one-time setup)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(LLAMA_CPP_DIR)],
        check=True,
    )
    if not script.exists():
        raise RuntimeError(f"Cloned llama.cpp but {script} still missing — repo layout may have changed.")
    return script


def _convert_to_gguf(merged_path: Path, version: str) -> Path:
    script = _ensure_llama_cpp()
    GGUF_DIR.mkdir(parents=True, exist_ok=True)
    out_file = GGUF_DIR / f"{version}.gguf"
    log.info("Converting merged model to GGUF (q8_0)...")
    subprocess.run(
        [sys.executable, str(script), str(merged_path), "--outfile", str(out_file), "--outtype", "q8_0"],
        check=True,
    )
    return out_file


# ── Ollama registration (HTTP API — no CLI dependency) ────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _upload_blob(client: httpx.Client, gguf_path: Path) -> str:
    """Upload the GGUF file as a content-addressed blob, keyed by its own
    sha256 digest — this is how Ollama's API imports a LOCAL file (the
    top-level 'from' field is for referencing an existing model tag by name,
    e.g. 'llama3.2', not a filesystem path — passing a Windows path there
    fails with 'invalid model name')."""
    digest = f"sha256:{_sha256_file(gguf_path)}"
    log.info("Uploading GGUF blob (%s)...", digest[:19])
    with gguf_path.open("rb") as f:
        resp = client.post(f"{OLLAMA_BASE}/api/blobs/{digest}", content=f.read())
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Blob upload failed: HTTP {resp.status_code} — {resp.text[:300]}")
    return digest


def _register_with_ollama(gguf_path: Path, tag: str) -> None:
    log.info("Registering %s with Ollama via /api/create...", tag)
    with httpx.Client(timeout=600.0) as client:
        digest = _upload_blob(client, gguf_path)
        with client.stream("POST", f"{OLLAMA_BASE}/api/create",
                            json={
                                "model": tag,
                                "files": {"model.gguf": digest},
                                "system": SYSTEM_PROMPT,
                                "parameters": {"temperature": 0.4},
                                "stream": True,
                            }) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                evt = json.loads(line)
                if "status" in evt:
                    log.info("ollama: %s", evt["status"])
                if evt.get("error"):
                    raise RuntimeError(f"Ollama create failed: {evt['error']}")


# ── Narrator version registry (mirrors train/registry.py's philosophy) ────

def _next_narrator_version() -> str:
    if not NARRATOR_REGISTRY.exists():
        return "v1"
    versions = json.loads(NARRATOR_REGISTRY.read_text())
    return f"v{len(versions) + 1}"


def _record_version(version: str, tag: str, dataset_path: Path, base_model: str, n_examples: int) -> None:
    versions = json.loads(NARRATOR_REGISTRY.read_text()) if NARRATOR_REGISTRY.exists() else []
    versions.append({
        "version": version,
        "ollama_tag": tag,
        "base_model": base_model,
        "dataset_path": str(dataset_path),
        "n_examples": n_examples,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    NARRATOR_REGISTRY.write_text(json.dumps(versions, indent=2))


def train_narrator(dataset_path: Path, epochs: int = 3, base_model: str = BASE_MODEL) -> str:
    examples = _load_dataset(dataset_path)
    if len(examples) < 4:
        raise ValueError(f"Only {len(examples)} examples in {dataset_path} — need at least a handful to train on.")

    version = _next_narrator_version()
    tag = f"{TAG_PREFIX}:{version}"
    output_dir = MERGED_DIR / version

    merged_path = _finetune(examples, output_dir, epochs, base_model)
    gguf_path = _convert_to_gguf(merged_path, version)
    _register_with_ollama(gguf_path, tag)
    _record_version(version, tag, dataset_path, base_model, len(examples))

    log.info("Done. Registered Ollama tag: %s", tag)
    return tag


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="LoRA fine-tune + GGUF-convert + register the SignalForge narrator.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a narrator/dataset/v*.jsonl file")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    args = parser.parse_args()

    tag = train_narrator(Path(args.dataset), epochs=args.epochs, base_model=args.base_model)
    print(f"Registered: {tag}")
