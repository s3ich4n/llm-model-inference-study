import multiprocessing
import os
import queue
import sys
from typing import Any

from llm.exceptions import UnexpectedResultTypeError, WorkerDiedError
from llm.model_worker import ModelWorker
from logs import logger

# Start method, per platform:
#
# Linux (incl. the Docker image) uses "fork": it is fast (copies the
# already-loaded torch parent instead of re-importing everything from
# scratch), and this project never has CUDA initialized in the parent
# process before the fork happens (the model only loads inside the worker
# child), so the "cannot re-initialize CUDA in forked subprocess" failure
# mode does not apply here.
#
# macOS must use "spawn". Our parent is multi-threaded (uvicorn's server
# threads, plus whatever torch/vLLM start at import time) and has already
# touched Apple's Objective-C runtime. fork() only carries the calling
# thread into the child, so any ObjC class whose +initialize was running on
# another thread at fork time is left half-initialized; the ObjC runtime
# detects this in the child and aborts it outright:
#
#   objc[...]: +[NSCharacterSet initialize] may have been in progress in
#   another thread when fork() was called. ... Crashing instead.
#
# The child dies before it ever reads the task queue, so every
# result_queue.get() below would block forever. "spawn" pays a slower
# startup for a child that is actually alive.
#
# "forkserver" also survives on macOS, but only with an empty preload list -
# and an empty preload list means the child re-imports torch/transformers
# from scratch anyway, i.e. exactly spawn's cost plus a long-lived extra
# process. Adding set_forkserver_preload([...]) - the only thing that makes
# forkserver cheaper than spawn - reintroduces the abort above, because the
# forkserver process is then the one holding half-initialized ObjC state
# when it forks. So: spawn.
#
# LLM_MP_START_METHOD overrides this, for experimenting.
_START_METHOD = os.environ.get(
    "LLM_MP_START_METHOD",
    "spawn" if sys.platform == "darwin" else "fork",
)
_mp_context = multiprocessing.get_context(_START_METHOD)

# How long to wait on the result queue before checking that the worker is
# still alive. Only bounds the liveness check, not total generation time.
_RESULT_POLL_SECONDS = 1.0


class ModelExecutor:
    def __init__(self):
        self.task_queue = _mp_context.Queue()
        self.result_queue = _mp_context.Queue()
        self.worker_process = None

        logger.debug("Model executor initialized (start method: %s)", _START_METHOD)

    def _wait_for_result(self) -> Any:
        """Block for a worker result, failing fast if the worker has died.

        A plain result_queue.get() waits forever when the worker process
        crashed (or was never able to start), which turns any such crash into
        a silent hang on the request path. Poll instead, and surface the exit
        code as soon as the process is gone.
        """
        while True:
            try:
                return self.result_queue.get(timeout=_RESULT_POLL_SECONDS)
            except queue.Empty:
                if self.worker_process is None:
                    raise WorkerDiedError("Worker process was never started") from None
                if not self.worker_process.is_alive():
                    # One last look: the worker may have queued its result and
                    # exited between the timeout and this check.
                    try:
                        return self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    raise WorkerDiedError(
                        "Worker process exited with code "
                        f"{self.worker_process.exitcode} before returning a result"
                    ) from None

    def setup_worker(self, model_name: str):
        logger.debug(f"Setting up worker with model: {model_name}")
        self.worker_process = _mp_context.Process(
            target=ModelWorker.run,
            args=(model_name, self.task_queue, self.result_queue),
            daemon=True,
        )
        logger.debug("Starting worker process")
        self.worker_process.start()
        logger.debug("Worker process started")

    def execute_batch(
        self,
        prompts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not prompts:
            logger.debug("Empty batch received")
            return []

        logger.debug(f"Sending batch to worker: {prompts}")
        # Send batch to worker
        self.task_queue.put((prompts, False))

        # Get results
        logger.debug("Waiting for results from worker")
        results = self._wait_for_result()
        logger.debug(f"Received results from worker: {results}")
        return results

    def execute_forward_batch(
        self,
        prompts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not prompts:
            logger.debug("Empty batch received")
            return []

        logger.debug(f"Sending streaming batch to worker: {prompts}")
        # Send batch to worker with streaming flag
        self.task_queue.put((prompts, True))

        # Get streaming results
        logger.debug("Waiting for streaming results from worker")
        result_type, results = self._wait_for_result()
        logger.debug(f"Received streaming results from worker: {results}")

        if result_type == 'stream':
            return results
        else:
            raise UnexpectedResultTypeError("Unexpected result type from worker")

    def shutdown(self, timeout: float = 30.0):
        if self.worker_process and self.worker_process.is_alive():
            logger.debug("Terminating worker process")
            self.worker_process.terminate()
            self.worker_process.join(timeout)

            if self.worker_process.is_alive():
                logger.warning(
                    "Worker process did not exit within %ss after SIGTERM; sending SIGKILL",
                    timeout,
                )
                self.worker_process.kill()
                self.worker_process.join()

            logger.debug("Worker process terminated")

    def __del__(self):
        self.shutdown()
