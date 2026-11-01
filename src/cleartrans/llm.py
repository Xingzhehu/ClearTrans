import os
import time
import requests


class ChatClient:
    """OpenAI-compatible HTTP transport; usage is measured, never estimated."""

    def __init__(self, config):
        self.config = config

    def generate(self, prompt, seed=42, temperature=None):
        cfg = self.config
        key = os.environ.get(cfg.get("api_key_env", "LLM_API_KEY"), "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": cfg["model"], "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": cfg.get("max_tokens", 4096), "seed": seed,
                   "temperature": cfg.get("temperature", 0.7) if temperature is None else temperature}
        for attempt in range(4):
            response = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions",
                                     headers=headers, json=payload, timeout=cfg.get("timeout", 600))
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 3:
                    time.sleep(min(2 ** attempt, 8))
                    continue
            if not response.ok:
                raise RuntimeError(f"LLM HTTP {response.status_code}; inspect service logs (credentials redacted)")
            data = response.json()
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RuntimeError("LLM output was truncated; increase max_tokens, do not accept partial code")
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("LLM returned no textual completion")
            return {"text": content, "usage": data.get("usage"), "model": data.get("model", cfg["model"]),
                    "finish_reason": choice.get("finish_reason"), "seed": seed}
        raise RuntimeError("LLM retry limit exceeded")
