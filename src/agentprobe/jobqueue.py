"""Reliable work queue on Redis with lease-based, at-least-once delivery.

Layout (all keys share a prefix)::

    <p>:pending      LIST  run ids waiting for a worker (LPUSH in, consumed from the right)
    <p>:processing   LIST  run ids a worker has taken but not acknowledged
    <p>:leases       HASH  run id -> unix deadline; a live worker keeps pushing it forward
    <p>:deliveries   HASH  run id -> how many times it has been handed out

``dequeue`` moves an id pending -> processing in one atomic Redis command (BLMOVE), so a worker
that dies right after taking a job can never lose it. If a worker stops heartbeating, ``reap``
sees the expired lease and puts the id back on the pending list.

A reaper only *sets* a missing lease (never requeues on it): the instant between BLMOVE and the
worker's first lease write is a legitimate state, and treating it as expiry would hand one job to
two workers. Duplicate delivery is still possible (at-least-once); the database claim in
``runner.claim_run`` is what makes that harmless.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class RedisQueue:
    def __init__(
        self,
        client,
        name: str = "agentprobe",
        lease_seconds: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.r = client
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.pending = f"{name}:pending"
        self.processing = f"{name}:processing"
        self.leases = f"{name}:leases"
        self.deliveries = f"{name}:deliveries"

    @classmethod
    def from_url(cls, url: str, **kwargs) -> RedisQueue:
        import redis  # imported lazily so the package works without Redis installed

        return cls(redis.Redis.from_url(url, decode_responses=True), **kwargs)

    def enqueue(self, run_id: int) -> None:
        self.r.lpush(self.pending, str(run_id))

    def dequeue(self, timeout: float = 1.0) -> tuple[int, int] | None:
        """Take one job. Returns (run_id, delivery_count) or None if nothing arrived in time."""
        raw = self.r.blmove(self.pending, self.processing, timeout, "RIGHT", "LEFT")
        if raw is None:
            return None
        self.r.hset(self.leases, raw, self.clock() + self.lease_seconds)
        deliveries = self.r.hincrby(self.deliveries, raw, 1)
        return int(raw), int(deliveries)

    def heartbeat(self, run_id: int) -> None:
        self.r.hset(self.leases, str(run_id), self.clock() + self.lease_seconds)

    def ack(self, run_id: int) -> None:
        key = str(run_id)
        pipe = self.r.pipeline()
        pipe.lrem(self.processing, 0, key)
        pipe.hdel(self.leases, key)
        pipe.hdel(self.deliveries, key)
        pipe.execute()

    def reap(self) -> list[int]:
        """Requeue jobs whose worker stopped heartbeating. Safe to call from every worker."""
        now = self.clock()
        requeued: list[int] = []
        for key in self.r.lrange(self.processing, 0, -1):
            deadline = self.r.hget(self.leases, key)
            if deadline is None:
                self.r.hset(self.leases, key, now + self.lease_seconds)
            elif float(deadline) < now and self.r.lrem(self.processing, 1, key) == 1:
                # lrem == 1 means this reaper won the race against any other reaper
                self.r.hdel(self.leases, key)
                self.r.rpush(self.pending, key)  # front of the consume side: retry soon
                requeued.append(int(key))
        return requeued

    def stats(self) -> dict[str, int]:
        return {"pending": self.r.llen(self.pending), "processing": self.r.llen(self.processing)}
