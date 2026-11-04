"""Chat model and embedding providers. Everything here runs on a free tier."""
import os
import time

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from core.logutil import log, Timer

_EMBEDDING = None


def get_embeddings():
    """FastEmbed (ONNX) singleton: loads in seconds on CPU, no torch."""
    global _EMBEDDING
    if _EMBEDDING is None:
        with Timer("embedding model load"):
            from langchain_community.embeddings import FastEmbedEmbeddings
            _EMBEDDING = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _EMBEDDING


PROVIDERS = {
    "Ollama (local, unlimited, no key)": {"model": "llama3.2", "env": "", "needs_key": False},
    "Pollinations (anonymous, rate-limited)": {
        "model": "openai", "env": "", "needs_key": False,
        "base_url": "https://text.pollinations.ai/openai",
    },
    "LM Studio (local server, no key)": {
        "model": "local-model", "env": "", "needs_key": False,
        "base_url": "http://localhost:1234/v1",
    },
    "llama.cpp (local server, no key)": {
        "model": "default", "env": "", "needs_key": False,
        "base_url": "http://localhost:8080/v1",
    },
    "Hugging Face (free tier)": {"model": "meta-llama/Llama-3.1-8B-Instruct", "env": "HF_TOKEN", "needs_key": True},
    "Groq (free tier: 30 rpm / 1k day)": {"model": "openai/gpt-oss-120b", "env": "GROQ_API_KEY", "needs_key": True},
    "Google Gemini (free Flash models)": {"model": "gemini-3-flash-preview", "env": "GEMINI_API_KEY", "needs_key": True},
    "OpenRouter (free models, 50/day)": {
        "model": "openai/gpt-oss-120b:free", "env": "OPENROUTER_API_KEY", "needs_key": True,
        "base_url": "https://openrouter.ai/api/v1",
    },
}

PROVIDER_LINKS = {
    "Groq (free tier: 30 rpm / 1k day)": "https://console.groq.com/keys",
    "Google Gemini (free Flash models)": "https://aistudio.google.com/app/apikey",
    "OpenRouter (free models, 50/day)": "https://openrouter.ai/keys",
    "Hugging Face (free tier)": "https://huggingface.co/settings/tokens",
    "Ollama (local, unlimited, no key)": "https://ollama.com/download",
    "LM Studio (local server, no key)": "https://lmstudio.ai",
    "llama.cpp (local server, no key)": "https://github.com/ggerganov/llama.cpp",
    "Pollinations (anonymous, rate-limited)": "https://pollinations.ai",
}


def ollama_status(model: str = "llama3.2") -> dict:
    import requests
    base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        r = requests.get(f"{base}/api/tags", timeout=2)
        models = [m.get("name", "") for m in r.json().get("models", [])]
        return {"running": True, "has_model": any(model in m for m in models), "models": models}
    except Exception:
        return {"running": False, "has_model": False, "models": []}


def resolve_provider() -> tuple:
    """Server-side config. Order: LLM_PROVIDER env (+ its key) -> working
    Ollama -> Pollinations fallback. Returns (provider_name, model, api_key)."""
    configured = os.getenv("LLM_PROVIDER", "Ollama (local, unlimited, no key)")
    if configured not in PROVIDERS:
        configured = "Ollama (local, unlimited, no key)"
    model = os.getenv("LLM_MODEL", "") or PROVIDERS[configured]["model"]

    if not configured.startswith("Ollama"):
        key = os.getenv(PROVIDERS[configured]["env"], "")
        if key:
            return configured, model, key

    status = ollama_status(model if configured.startswith("Ollama") else PROVIDERS["Ollama (local, unlimited, no key)"]["model"])
    if status["running"] and status["has_model"]:
        name = "Ollama (local, unlimited, no key)"
        return name, PROVIDERS[name]["model"], ""

    return (
        "Pollinations (anonymous, rate-limited)",
        PROVIDERS["Pollinations (anonymous, rate-limited)"]["model"],
        "",
    )


def get_llm(provider: str, api_key: str, model: str | None = None):
    if PROVIDERS[provider]["needs_key"] and not api_key:
        raise ValueError(f"{provider} needs an API key. Get a free one at {PROVIDER_LINKS[provider]}")
    model = model or PROVIDERS[provider]["model"]

    if provider.startswith("Ollama"):
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model, temperature=0.2,
            base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )

    if provider.startswith("Groq"):
        from langchain_groq import ChatGroq
        return ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.2)

    if provider.startswith("Google Gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.2)

    if provider.startswith("Hugging Face"):
        return HuggingFaceChat(model=model, api_key=api_key, temperature=0.2)

    if "base_url" in PROVIDERS[provider]:
        return OpenAICompatibleChat(
            model=model, api_key=api_key, temperature=0.2,
            base_url=PROVIDERS[provider]["base_url"],
        )

    raise ValueError(f"Unsupported provider: {provider}")


class OpenAICompatibleChat(BaseChatModel):
    """Any OpenAI-compatible endpoint: OpenRouter, Pollinations, LM Studio,
    llama.cpp, vLLM. Plain requests, no extra dependencies."""

    model: str
    api_key: str = ""
    temperature: float = 0.2
    base_url: str = ""

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def _to_openai_role(self, m) -> str:
        return {"human": "user", "ai": "assistant"}.get(m.type, m.type)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        import requests
        payload = {
            "model": self.model,
            "messages": [
                {"role": self._to_openai_role(m), "content": m.content if isinstance(m.content, str) else str(m.content)}
                for m in messages
            ],
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=120,
                )
            except Exception as e:
                last_error = f"connection failed (attempt {attempt + 1}/3): {e}"
                log.warning(last_error)
                time.sleep(2 ** attempt)
                continue
            if resp.status_code == 200:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=resp.json()["choices"][0]["message"]["content"]))]
                )
            last_error = f"LLM error {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in (429, 500, 502, 503) and attempt < 2:
                log.warning(f"{last_error} — retrying")
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(last_error)
        raise RuntimeError(last_error)


class HuggingFaceChat(BaseChatModel):
    """Hugging Face Inference API with a free account token."""

    model: str
    api_key: str
    temperature: float = 0.2

    @property
    def _llm_type(self) -> str:
        return "huggingface"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from huggingface_hub import InferenceClient
        from langchain_core.messages import SystemMessage
        client = InferenceClient(token=self.api_key)
        out = client.chat.completions.create(
            model=self.model,
            messages=[
                ({"role": "system" if isinstance(m, SystemMessage) else m.type, "content": str(m.content)})
                for m in messages
            ],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=out.choices[0].message.content))]
        )
