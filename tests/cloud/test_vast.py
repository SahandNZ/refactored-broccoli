import pytest

from src.cloud import _vast


def test_destroy_answers_the_prompt_and_waits_until_gone(monkeypatch):
    calls = []
    listed = [[{"id": 7}], []]

    def fake(*arguments, answer=""):
        calls.append((arguments, answer))
        return listed.pop(0) if arguments[:2] == ("show", "instances") else "destroying"

    monkeypatch.setattr(_vast, "_vastai", fake)
    monkeypatch.setattr(_vast.time, "sleep", lambda _: None)
    _vast.destroy_instance(7, timeout_seconds=60)
    destroys = [answer for arguments, answer in calls if arguments[0] == "destroy"]
    assert destroys == ["y\n", "y\n"]


def test_destroy_raises_when_the_instance_stays(monkeypatch):
    monkeypatch.setattr(
        _vast,
        "_vastai",
        lambda *arguments, answer="": [{"id": 7}] if arguments[0] == "show" else "",
    )
    monkeypatch.setattr(_vast.time, "sleep", lambda _: None)
    clock = iter(range(0, 10_000, 50))
    monkeypatch.setattr(_vast.time, "monotonic", lambda: next(clock))
    with pytest.raises(RuntimeError, match="still listed"):
        _vast.destroy_instance(7, timeout_seconds=120)


def test_cheapest_offer_reads_the_first_offer(monkeypatch):
    offer = {
        "id": 5,
        "gpu_name": "RTX 3060",
        "gpu_ram": 12288,
        "dph_total": 0.05,
        "geolocation": "PL",
    }
    monkeypatch.setattr(_vast, "_vastai", lambda *arguments, answer="": [offer])
    chosen = _vast.cheapest_offer(max_price=0.15, min_vram_gb=12, min_cuda=12.8)
    assert (chosen.id, chosen.vram_gb, chosen.price_per_hour) == (5, 12.0, 0.05)
    monkeypatch.setattr(_vast, "_vastai", lambda *arguments, answer="": [])
    with pytest.raises(LookupError):
        _vast.cheapest_offer(max_price=0.01, min_vram_gb=12, min_cuda=12.8)
