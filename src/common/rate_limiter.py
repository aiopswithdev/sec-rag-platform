import os
import time
import threading
import logging

logger = logging.getLogger("TPMLimiter")

class TPMLimiter:
    """Sliding-window token-per-minute gate. Thread-safe so sync LangGraph nodes can call it directly."""
    def __init__(self, tpm_budget: int = 7000, window_s: float = 60.0):
        self.tpm_budget = tpm_budget
        self.window_s = window_s
        self._usage: list[list] = []  # [timestamp, tokens] — mutable so reconcile() can adjust
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int) -> float:
        """Reserve budget. Returns the reservation timestamp, used as a handle for reconcile()."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._usage = [u for u in self._usage if now - u[0] < self.window_s]
                current_total = sum(u[1] for u in self._usage)
                if current_total + estimated_tokens <= self.tpm_budget:
                    self._usage.append([now, estimated_tokens])
                    return now
                wait_time = max(self.window_s - (now - self._usage[0][0]), 1.0)
            logger.info(f"Pacing TPM: limit reached ({current_total}/{self.tpm_budget}). Sleeping {wait_time:.1f}s...")
            time.sleep(wait_time)  # sleep OUTSIDE the lock

    def reconcile(self, handle: float, actual_tokens: int):
        """Replace an estimate with real usage from the API response. Estimates deliberately
        over-reserve (full max_tokens completion budget); without this, unused reservations
        pile up and trigger long, unnecessary sleeps that stale the HTTP connection pool."""
        with self._lock:
            for entry in self._usage:
                if entry[0] == handle:
                    entry[1] = max(int(actual_tokens), 0)
                    return


planner_limiter = TPMLimiter(tpm_budget=int(os.getenv("PLANNER_TPM_BUDGET", "7000")))
synthesis_limiter = TPMLimiter(tpm_budget=int(os.getenv("SYNTHESIS_TPM_BUDGET", "7000")))