import multiprocessing
from typing import Any

from llm.exceptions import UnexpectedResultTypeError
from llm.model_worker import ModelWorker
from logs import logger

# Deliberately "fork", not "spawn": fork is fast (copies the already-loaded
# torch/CUDA-initialized parent instead of re-importing everything from
# scratch), and this project never has CUDA initialized in the parent process
# before the fork happens (the model only loads inside the worker child), so
# the "cannot re-initialize CUDA in forked subprocess" failure mode does not
# apply here. This does still trip Python's fork-safety DeprecationWarning
# when the parent is multi-threaded (e.g. under pytest) - that's a known,
# separate, currently-accepted tradeoff, not a bug.
_mp_context = multiprocessing.get_context("fork")


class ModelExecutor:
    def __init__(self):
        self.task_queue = _mp_context.Queue()
        self.result_queue = _mp_context.Queue()
        self.worker_process = None

        logger.debug("Model executor initialized")

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
        results = self.result_queue.get()
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
        result_type, results = self.result_queue.get()
        logger.debug(f"Received streaming results from worker: {results}")

        if result_type == 'stream':
            return results
        else:
            raise UnexpectedResultTypeError("Unexpected result type from worker")

    def shutdown(self, timeout: float = 5.0):
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
