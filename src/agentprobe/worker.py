"""Queue worker: pulls run ids from Redis and executes them.

Run as many of these as you like, on any machines that can reach Redis and the database::

    agentprobe worker --redis-url redis://localhost:6379/0
"""

from __future__ import annotations

import logging
import threading

from .db import Database
from .jobqueue import RedisQueue
from .runner import execute_run, fail_run, release_run

log = logging.getLogger("agentprobe.worker")


class Worker:
    def __init__(
        self,
        db: Database,
        queue: RedisQueue,
        poll_timeout: float = 1.0,
        max_deliveries: int = 3,
    ) -> None:
        self.db, self.queue = db, queue
        self.poll_timeout = poll_timeout
        self.max_deliveries = max_deliveries

    def run_once(self) -> bool:
        """Reap dead leases, then handle at most one job. True if a job was taken."""
        for run_id in self.queue.reap():
            log.warning("lease expired, requeueing run %s", run_id)
            release_run(self.db, run_id)  # so the next worker can claim it again
        job = self.queue.dequeue(self.poll_timeout)
        if job is None:
            return False
        run_id, deliveries = job
        if deliveries > self.max_deliveries:
            # a run that keeps killing its workers must not loop forever
            fail_run(self.db, run_id, f"abandoned after {deliveries - 1} failed deliveries")
            self.queue.ack(run_id)
            return True
        self._process(run_id)
        return True

    def _process(self, run_id: int) -> None:
        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(run_id, stop), daemon=True)
        beat.start()
        try:
            executed = execute_run(self.db, run_id)
            if not executed:
                log.info("run %s already owned or finished, skipping", run_id)
        finally:
            stop.set()
            beat.join(timeout=5)
        # Reached only if execute_run returned normally; an exception or a killed process
        # skips the ack, the lease lapses, and the run is redelivered.
        self.queue.ack(run_id)

    def _heartbeat(self, run_id: int, stop: threading.Event) -> None:
        interval = max(0.05, self.queue.lease_seconds / 3)
        while not stop.wait(interval):
            try:
                self.queue.heartbeat(run_id)
            except Exception:
                log.exception("heartbeat failed for run %s", run_id)

    def run_forever(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        log.info("worker started")
        while not stop.is_set():
            try:
                self.run_once()
            except Exception:
                log.exception("worker loop error")
                stop.wait(1.0)
        log.info("worker stopped")
