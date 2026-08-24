import os
import time
import asyncio
import logging

logger = logging.getLogger("TPMLimiter")

class TPMLimiter:
    """Sliding-window token-per-minute gate shared across LLM calls."""
    def __init__(self, tpm_budget: int = 7000, window_s: float = 60.0):
        self.tpm_budget = tpm_budget
        self.window_s = window_s
        self._usage: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int):
        async with self._lock:
            now = time.monotonic()
            # Prune records older than window_s
            self._usage = [(t, n) for t, n in self._usage if now - t < self.window_s]
            current_total = sum(n for _, n in self._usage)
            
            if current_total + estimated_tokens > self.tpm_budget:
                wait_time = max(self.window_s - (now - self._usage[0][0]), 1.0)
                logger.info(f"Pacing TPM: limit reached ({current_total}/{self.tpm_budget}). Sleeping {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
                return await self.acquire(estimated_tokens)
                
            self._usage.append((now, estimated_tokens))

groq_limiter = TPMLimiter(tpm_budget=int(os.getenv("GROQ_TPM_BUDGET", "7000")))