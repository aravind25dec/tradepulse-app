"""
backend/whatsapp_notifier.py
Shared WhatsApp sender for all TradePulse apps, via the Meta WhatsApp Cloud API.

Requires in backend/.env:
  WHATSAPP_ACCESS_TOKEN   - permanent or temporary token from the Meta app
  WHATSAPP_PHONE_NUMBER_ID - the "from" number's phone_number_id (not the phone number itself)
  WHATSAPP_RECIPIENT       - recipient phone number in E.164 format, e.g. 15551234567 (no '+')

Docs: https://developers.facebook.com/docs/whatsapp/cloud-api/reference/messages
"""

import logging
import os

import httpx

log = logging.getLogger(__name__)

GRAPH_API_VERSION = "v20.0"


def configured() -> bool:
    return bool(
        os.getenv("WHATSAPP_ACCESS_TOKEN")
        and os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        and os.getenv("WHATSAPP_RECIPIENT")
    )


async def send_whatsapp_message(body: str) -> dict:
    """Send a plain-text WhatsApp message via the Meta Cloud API.

    Returns {"success": True, ...} or {"success": False, "error": "..."}.
    """
    token      = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    phone_id   = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    recipient  = os.getenv("WHATSAPP_RECIPIENT", "")

    if not (token and phone_id and recipient):
        return {"success": False, "error": "WhatsApp not configured (missing env vars)"}

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{phone_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient,
        "type": "text",
        "text": {"body": body[:4096]},
    }
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            log.error("WhatsApp send failed: %s %s", resp.status_code, resp.text)
            return {"success": False, "error": f"{resp.status_code}: {resp.text}"}
        return {"success": True, "response": resp.json()}
    except Exception as exc:
        log.error("WhatsApp send error: %s", exc)
        return {"success": False, "error": str(exc)}
