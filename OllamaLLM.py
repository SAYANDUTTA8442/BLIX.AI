import json
import time
import requests
from typing import Generator, Optional


class OllamaLLM:
    def __init__(
        self,
        model: str      = "mistral",
        base_url: str   = "http://localhost:11434",
        timeout: int    = 60,
        retries: int    = 3,
        backoff: float  = 2.0,   # seconds — doubles each retry
        system: str     = "",    # optional system prompt for every call
    ):
        self.model   = model
        self.base    = base_url.rstrip("/")
        self.url     = f"{self.base}/api/generate"
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.system  = system

        # Cumulative token usage across all calls in this session
        self._total_prompt_tokens   = 0
        self._total_eval_tokens     = 0

    # =========================
    # HEALTH CHECK
    # =========================
    def is_available(self) -> bool:
        """Ping Ollama server. Returns True if reachable and model is loaded."""
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            # Accept both  "mistral"  and  "mistral:latest"
            return any(m.split(":")[0] == self.model.split(":")[0] for m in models)
        except requests.exceptions.RequestException:
            return False

    # =========================
    # BUILD PAYLOAD
    # =========================
    def _payload(self, prompt: str, temperature: float, stream: bool) -> dict:
        payload: dict = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  stream,
            "options": {"temperature": temperature},
        }
        if self.system:
            payload["system"] = self.system
        return payload

    # =========================
    # MAIN GENERATE
    # =========================
    def generate(
        self,
        prompt:      str,
        temperature: float           = 0.3,
        raw:         bool            = False,   # True → return full response dict
    ) -> str | dict:
        """
        Send a prompt and return the model's text response.
        Retries with exponential back-off on transient errors.
        Raises RuntimeError after all retries are exhausted.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        payload = self._payload(prompt, temperature, stream=False)
        delay   = self.backoff

        for attempt in range(1, self.retries + 1):
            try:
                response = requests.post(self.url, json=payload, timeout=self.timeout)

                if response.status_code == 200:
                    data = response.json()
                    self._track_tokens(data)
                    return data if raw else data.get("response", "")

                # Non-200 — surface the Ollama error message clearly
                try:
                    err_msg = response.json().get("error", response.text)
                except Exception:
                    err_msg = response.text
                raise RuntimeError(f"Ollama HTTP {response.status_code}: {err_msg}")

            except requests.exceptions.ConnectionError:
                print(f"  [Attempt {attempt}/{self.retries}] Connection refused — is Ollama running?")
            except requests.exceptions.Timeout:
                print(f"  [Attempt {attempt}/{self.retries}] Request timed out after {self.timeout}s")
            except RuntimeError:
                raise   # Don't retry on a clean Ollama error
            except requests.exceptions.RequestException as e:
                print(f"  [Attempt {attempt}/{self.retries}] Request error: {e}")

            if attempt < self.retries:
                print(f"  ⏳ Retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay *= 2   # exponential back-off

        raise RuntimeError(
            f"❌ Ollama failed after {self.retries} attempts. "
            f"Check that the server is up and '{self.model}' is pulled."
        )

    # =========================
    # STREAMING GENERATE
    # =========================
    def generate_stream(
        self,
        prompt:      str,
        temperature: float = 0.3,
    ) -> Generator[str, None, None]:
        """
        Stream the response token-by-token.
        Yields text chunks as they arrive.
        Raises RuntimeError on connection / parse failure.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        payload = self._payload(prompt, temperature, stream=True)

        try:
            with requests.post(
                self.url, json=payload, stream=True, timeout=self.timeout
            ) as response:
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Ollama HTTP {response.status_code}: {response.text}"
                    )
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line.decode("utf-8"))
                    except json.JSONDecodeError:
                        continue   # skip malformed lines

                    token = chunk.get("response", "")
                    if token:
                        yield token

                    # Final chunk carries usage stats
                    if chunk.get("done"):
                        self._track_tokens(chunk)
                        break

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"[STREAM ERROR]: {e}") from e

    # =========================
    # TOKEN TRACKING
    # =========================
    def _track_tokens(self, data: dict) -> None:
        self._total_prompt_tokens += data.get("prompt_eval_count", 0)
        self._total_eval_tokens   += data.get("eval_count", 0)

    @property
    def token_usage(self) -> dict:
        """Cumulative token usage for this session."""
        return {
            "prompt_tokens":     self._total_prompt_tokens,
            "completion_tokens": self._total_eval_tokens,
            "total_tokens":      self._total_prompt_tokens + self._total_eval_tokens,
        }

    def reset_token_usage(self) -> None:
        self._total_prompt_tokens = 0
        self._total_eval_tokens   = 0

    # =========================
    # REPR
    # =========================
    def __repr__(self) -> str:
        return (
            f"OllamaLLM(model={self.model!r}, url={self.url!r}, "
            f"timeout={self.timeout}, retries={self.retries})"
        )


# =========================
# TEST
# =========================
if __name__ == "__main__":
    llm = OllamaLLM(model="mistral")

    # --- Health check before any work ---
    print("🩺 Checking Ollama availability...")
    if not llm.is_available():
        print(f"❌ Ollama is not reachable or '{llm.model}' is not pulled.")
        print("   Run:  ollama pull mistral")
        raise SystemExit(1)
    print("✅ Ollama is ready.\n")

    # --- Standard generate ---
    prompt = "Extract topics from an Operating Systems syllabus in bullet points."
    print("🧠 Response (standard):\n")
    result = llm.generate(prompt)
    print(result)

    # --- Streaming generate ---
    print("\n🌊 Response (streaming):\n")
    for chunk in llm.generate_stream(prompt):
        print(chunk, end="", flush=True)
    print()

    # --- Token usage ---
    print(f"\n📈 Token usage: {llm.token_usage}")