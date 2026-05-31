from infergate.policies import infergate_score, make_policy
from infergate.schemas import Action, CacheState, LoadSnapshot, PolicySettings, QueueState, RequestContext


def make_request(**kwargs) -> RequestContext:
    defaults = dict(
        cache_key="k",
        estimated_cost=200,
        prompt_tokens=100,
        max_tokens=100,
        prompt_text="hello",
    )
    defaults.update(kwargs)
    return RequestContext(**defaults)


def test_infergate_score_uses_session_progress_and_cost() -> None:
    load = LoadSnapshot()
    queue = QueueState()
    cache = CacheState(cache_key="k")
    early = make_request(session_step=0, session_total_steps=5, utility=1.0, estimated_cost=200)
    late = make_request(session_step=4, session_total_steps=5, utility=1.0, estimated_cost=200)
    costly = make_request(session_step=4, session_total_steps=5, utility=1.0, estimated_cost=1000)
    assert infergate_score(late, load, queue, cache, include_cache=False) > infergate_score(
        early, load, queue, cache, include_cache=False
    )
    assert infergate_score(costly, load, queue, cache, include_cache=False) < infergate_score(
        late, load, queue, cache, include_cache=False
    )


def test_static_threshold_rejects_high_kv() -> None:
    policy = make_policy("static_threshold", PolicySettings(kv_reject_threshold=0.8))
    decision = policy.decide(make_request(), LoadSnapshot(kv_cache_usage_perc=0.9), QueueState(), CacheState(cache_key="k"))
    assert decision.action == Action.REJECT
    assert decision.reason == "kv_threshold"


def test_sjf_prioritizes_smaller_cost() -> None:
    policy = make_policy("sjf")
    load = LoadSnapshot(num_requests_waiting=1)
    queue = QueueState()
    cache = CacheState(cache_key="k")
    small = policy.decide(make_request(estimated_cost=100), load, queue, cache)
    large = policy.decide(make_request(estimated_cost=1000), load, queue, cache)
    assert small.priority > large.priority
    assert small.action == Action.DEFER


def test_edf_deadline_requests_rank_ahead_of_no_deadline() -> None:
    policy = make_policy("edf")
    load = LoadSnapshot(num_requests_waiting=1)
    queue = QueueState()
    cache = CacheState(cache_key="k")
    with_deadline = policy.decide(make_request(deadline_ms=500), load, queue, cache)
    no_deadline = policy.decide(make_request(deadline_ms=None), load, queue, cache)
    assert with_deadline.priority > no_deadline.priority


def test_vtc_inspired_rejects_over_limit_under_overload() -> None:
    policy = make_policy("vtc_inspired", PolicySettings(vtc_tenant_token_limit=100))
    request = make_request(tenant_id="tenant-a")
    queue = QueueState(tenant_token_debt={"tenant-a": 200})
    load = LoadSnapshot(num_requests_waiting=1)
    decision = policy.decide(request, load, queue, CacheState(cache_key="k"))
    assert decision.action == Action.REJECT
    assert "tenant" in decision.reason


def test_infergate_cache_boosts_reuse_priority() -> None:
    policy = make_policy("infergate_cache")
    request = make_request(utility=1.0, estimated_cost=500)
    load = LoadSnapshot(num_requests_waiting=1, warmup_allowed=True)
    queue = QueueState()
    cold = policy.decide(request, load, queue, CacheState(cache_key="k", predicted_reuse_count=0))
    hot = policy.decide(request, load, queue, CacheState(cache_key="k", predicted_reuse_count=4))
    assert hot.score > cold.score
    assert hot.warmup_candidate is True

