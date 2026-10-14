"""Optional LLM integration supporting Groq (fast inference) and OpenAI (gpt-4o-mini).

If GROQ_API_KEY or OPENAI_API_KEY is defined in .env, this module generates a fluent,
natural-language response while strictly preserving the deterministic calculations,
conflict warnings, refusal policies, and document citations from the core assistant.
If no key is configured, the system operates in offline rule-based mode.
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()


def get_llm_config() -> dict:
    """Detect configured LLM provider and credentials."""
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    configured_model = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b").strip()

    if groq_key:
        model = configured_model if configured_model else "llama-3.3-70b-versatile"
        return {
            "enabled": True,
            "provider": "Groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key": groq_key,
            "model": model,
        }
    elif openai_key:
        return {
            "enabled": True,
            "provider": "OpenAI",
            "url": "https://api.openai.com/v1/chat/completions",
            "key": openai_key,
            "model": configured_model or "gpt-4o-mini",
        }
    return {
        "enabled": False,
        "provider": "Offline",
        "url": "",
        "key": "",
        "model": "Rule-based & Decimal Calculator",
    }


def generate_llm_answer(question: str, answer_obj) -> str | None:
    """Generate a fluent, grounded response using Groq / OpenAI based on verified answer data."""
    cfg = get_llm_config()
    if not cfg["enabled"]:
        return None

    system_prompt = (
        "You are the senior MGC Developments Sales AI Assistant for MGC Aurora Heights (GT Road, Islamabad).\n"
        "Your task is to present the verified document facts clearly to sales staff.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Never guess or hallucinate numbers. Use the EXACT calculated figures and steps provided below.\n"
        "2. If the status is 'conflict' (e.g. transfer fee), clearly state that documents disagree and advise verifying with management.\n"
        "3. If the status is 'refused' (e.g. rental yield or bank rates) or 'unconfirmed' (e.g. anchor tenant, Sui gas), "
        "politely refuse to quote and direct the customer as instructed.\n"
        "4. Always mention the exact document name(s) and section(s) provided in the sources."
    )

    context_payload = (
        f"Salesperson Question: {question}\n\n"
        f"Verified Assistant Status: {answer_obj.status}\n"
        f"Headline / Result: {answer_obj.headline}\n"
        f"Calculation & Details: {chr(10).join(answer_obj.details) if answer_obj.details else 'N/A'}\n"
        f"Policy Notes: {chr(10).join(answer_obj.notes) if answer_obj.notes else 'N/A'}\n"
        f"Sources: {', '.join(answer_obj.sources) if answer_obj.sources else 'None'}"
    )

    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_payload},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }

    try:
        resp = requests.post(
            cfg["url"],
            headers={
                "Authorization": f"Bearer {cfg['key']}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            return f"[LLM Error {resp.status_code} ({cfg['provider']}): {resp.text[:200]}]"
    except Exception as e:
        return f"[LLM Connection Error: {str(e)}]"
