"""Tests for async event bus bridging sync engine events to async subscribers."""

import asyncio
import json
import threading

import pytest

from optimization_copilot.platform.events import AsyncEventBus


# ── Helpers ────────────────────────────────────────────────────────


def _run_async(coro):
    """Run a coroutine in a new event loop (for tests not using pytest-asyncio)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Subscribe / Unsubscribe ────────────────────────────────────────


class TestSubscription:
    def test_subscribe_returns_asyncio_queue(self):
        bus = AsyncEventBus()
        queue = bus.subscribe(campaign_id="abc")
        assert isinstance(queue, asyncio.Queue)

    def test_subscriber_count_increments(self):
        bus = AsyncEventBus()
        assert bus.subscriber_count("abc") == 0
        bus.subscribe(campaign_id="abc")
        assert bus.subscriber_count("abc") == 1

    def test_unsubscribe_removes_queue(self):
        bus = AsyncEventBus()
        queue = bus.subscribe(campaign_id="abc")
        bus.unsubscribe(queue)
        assert bus.subscriber_count("abc") == 0

    def test_unsubscribe_nonexistent_queue_does_not_error(self):
        bus = AsyncEventBus()
        fake_queue: asyncio.Queue = asyncio.Queue()
        bus.unsubscribe(fake_queue)  # should not raise

    def test_subscribe_unsubscribe_publish_cycle(self):
        bus = AsyncEventBus()
        q = bus.subscribe(campaign_id="test")
        bus.unsubscribe(q)
        # Publishing after unsubscribe should not fail or deliver
        bus.publish("test", "ping", {"v": 1})
        assert bus.subscriber_count("test") == 0


# ── Publish / Dispatch ─────────────────────────────────────────────


class TestPublish:
    def test_publish_sends_to_campaign_subscriber(self):
        bus = AsyncEventBus()
        queue = bus.subscribe(campaign_id="camp1")
        bus.publish("camp1", "iteration_complete", {"iteration": 5})

        async def _get():
            return queue.get_nowait()

        msg = _run_async(_get())
        assert msg["campaign_id"] == "camp1"
        assert msg["event"] == "iteration_complete"
        assert msg["data"]["iteration"] == 5
        assert "timestamp" in msg

    def test_publish_sends_to_global_subscriber(self):
        bus = AsyncEventBus()
        global_queue = bus.subscribe(campaign_id=None)
        bus.publish("camp-x", "started", {"info": "go"})

        msg = global_queue.get_nowait()
        assert msg["campaign_id"] == "camp-x"
        assert msg["event"] == "started"

    def test_publish_does_not_send_to_unrelated_campaign(self):
        bus = AsyncEventBus()
        q_camp_a = bus.subscribe(campaign_id="camp-a")
        bus.subscribe(campaign_id="camp-b")  # subscribe but don't track
        bus.publish("camp-b", "done", {"ok": True})

        assert q_camp_a.empty(), "camp-a subscriber should not receive camp-b events"

    def test_multiple_subscribers_receive_same_event(self):
        bus = AsyncEventBus()
        q1 = bus.subscribe(campaign_id="shared")
        q2 = bus.subscribe(campaign_id="shared")
        bus.publish("shared", "update", {"val": 42})

        msg1 = q1.get_nowait()
        msg2 = q2.get_nowait()
        assert msg1["data"]["val"] == 42
        assert msg2["data"]["val"] == 42

    def test_empty_subscribers_list_does_not_error(self):
        bus = AsyncEventBus()
        # No subscribers at all
        bus.publish("nobody-listens", "ghost_event", {"x": 1})

    def test_event_data_structure(self):
        bus = AsyncEventBus()
        q = bus.subscribe(campaign_id="c1")
        bus.publish("c1", "my_event", {"key": "value"})

        msg = q.get_nowait()
        assert set(msg.keys()) == {"campaign_id", "event", "data", "timestamp"}
        assert isinstance(msg["timestamp"], float)

    def test_specific_campaign_subscriber_ignores_other_campaigns(self):
        bus = AsyncEventBus()
        q_only_a = bus.subscribe(campaign_id="alpha")
        bus.publish("beta", "some_event", {})
        assert q_only_a.empty()


# ── Queue Overflow ─────────────────────────────────────────────────


class TestQueueOverflow:
    def test_queue_overflow_drops_oldest(self):
        bus = AsyncEventBus()
        q = bus.subscribe(campaign_id="overflow", maxsize=2)

        bus.publish("overflow", "e1", {"seq": 1})
        bus.publish("overflow", "e2", {"seq": 2})
        bus.publish("overflow", "e3", {"seq": 3})  # should drop e1

        msg_a = q.get_nowait()
        msg_b = q.get_nowait()
        # After overflow, oldest (e1) was dropped; we should get e2 and e3
        assert msg_a["data"]["seq"] == 2
        assert msg_b["data"]["seq"] == 3


# ── Thread Safety ──────────────────────────────────────────────────


class TestThreadSafety:
    def test_publish_from_non_async_thread(self):
        """Verify publish works when called from a background thread (no loop set)."""
        bus = AsyncEventBus()  # no loop
        q = bus.subscribe(campaign_id="threaded")
        errors = []

        def background_publish():
            try:
                bus.publish("threaded", "bg_event", {"from": "thread"})
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=background_publish)
        t.start()
        t.join()

        assert errors == []
        msg = q.get_nowait()
        assert msg["event"] == "bg_event"

    def test_publish_with_loop_set(self):
        """Verify publish dispatches via call_soon_threadsafe when loop is running."""
        bus = AsyncEventBus()
        q = bus.subscribe(campaign_id="loop-test")

        async def _test():
            bus.loop = asyncio.get_event_loop()
            bus.publish("loop-test", "async_event", {"v": 99})
            # Give the event loop a chance to process
            await asyncio.sleep(0.01)
            return q.get_nowait()

        msg = _run_async(_test())
        assert msg["event"] == "async_event"
        assert msg["data"]["v"] == 99


# ── SSE Formatting ─────────────────────────────────────────────────


class TestFormatSSE:
    def test_format_sse_returns_proper_sse_string(self):
        msg = {"event": "test", "data": {"x": 1}}
        sse = AsyncEventBus.format_sse(msg)
        assert sse.startswith("data: ")
        assert sse.endswith("\n\n")
        # The payload should be valid JSON
        payload = sse[len("data: "):-2]
        parsed = json.loads(payload)
        assert parsed == msg

    def test_format_sse_with_nested_data(self):
        msg = {"campaign_id": "c1", "event": "update", "data": {"nested": {"a": [1, 2]}}}
        sse = AsyncEventBus.format_sse(msg)
        payload = json.loads(sse[len("data: "):-2])
        assert payload["data"]["nested"]["a"] == [1, 2]


# ── Clear ──────────────────────────────────────────────────────────


class TestClear:
    def test_clear_removes_all_subscribers(self):
        bus = AsyncEventBus()
        bus.subscribe(campaign_id="a")
        bus.subscribe(campaign_id="b")
        bus.subscribe(campaign_id=None)
        bus.clear()
        assert bus.subscriber_count("a") == 0
        assert bus.subscriber_count("b") == 0
        assert bus.subscriber_count(None) == 0
