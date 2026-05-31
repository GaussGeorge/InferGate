import types

from infergate.tokenizer import TokenEstimator, build_request_context, extract_prompt_text, prompt_prefix_hash


def test_extract_prompt_text_and_cost_defaults() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Explain cache reuse."},
        ],
        "max_tokens": 32,
    }
    estimator = TokenEstimator()
    prompt = extract_prompt_text(payload)
    ctx = build_request_context(payload, {}, estimator)
    assert "system:" in prompt
    assert ctx.utility == 1.0
    assert ctx.session_step == 0
    assert ctx.deadline_ms is None
    assert ctx.max_tokens == 32
    assert ctx.estimated_cost == ctx.prompt_tokens + 32
    assert ctx.cache_key == prompt_prefix_hash(prompt)
    assert ctx.tokenizer_fallback is True


def test_metadata_and_headers_override_defaults() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {
            "x-request-utility": 3.5,
            "x-session-total-steps": 4,
            "x-cache-key": "meta-key",
        },
    }
    headers = {
        "x-session-id": "s1",
        "x-session-step": "2",
        "x-request-deadline-ms": "5000",
        "x-cache-key": "header-key",
    }
    ctx = build_request_context(payload, headers, TokenEstimator())
    assert ctx.session_id == "s1"
    assert ctx.session_step == 2
    assert ctx.session_total_steps == 4
    assert ctx.utility == 3.5
    assert ctx.deadline_ms == 5000
    assert ctx.cache_key == "header-key"


def test_tokenizer_id_does_not_default_to_model_id(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ID", "qwen")
    monkeypatch.delenv("INFERGATE_TOKENIZER_ID", raising=False)
    estimator = TokenEstimator(prefer_hf=True)
    assert estimator.tokenizer_id is None
    assert estimator.fallback is True


def test_tokenizer_id_uses_dedicated_env(monkeypatch) -> None:
    loaded: dict[str, str] = {}

    class FakeTokenizer:
        def encode(self, text: str) -> list[int]:
            return [1, 2, 3]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id: str) -> FakeTokenizer:
            loaded["tokenizer_id"] = tokenizer_id
            return FakeTokenizer()

    monkeypatch.setenv("MODEL_ID", "qwen")
    monkeypatch.setenv("INFERGATE_TOKENIZER_ID", r"D:\model_path\qwen3.5-4b")
    monkeypatch.setitem(
        __import__("sys").modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )
    estimator = TokenEstimator(prefer_hf=True)
    assert loaded["tokenizer_id"] == r"D:\model_path\qwen3.5-4b"
    assert estimator.fallback is False
    assert estimator.count("hello") == 3
