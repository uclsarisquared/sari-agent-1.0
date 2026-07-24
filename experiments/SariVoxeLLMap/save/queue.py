"""
Universal queue for saving plots and other files,
so that the main process will focus on actual processing and not be blocked by file I/O.

Due to this problem:
- Even though the processing and saving successfully occur in separate threads in the voxelization submodule,
this only occurs within each frame processing.
- Therefore, for the next frame to be processed, the previous frame's overall processing (including saving) must be completed.
- To combat this, there should be a single universal queue for saving files, so that the main process can focus on processing frames and not be blocked by file I/O.
"""

import asyncio, logging, os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Import saving functions
"""
To return:
- pointcloud
    - inputs
        - extrinsics
        - intrinsics
    - outputs
        - depth [shape]W
    - pointcloud (create new)
        - *points
        - colors
- voxelization
    - preprocessed points
    - no-outlier points
    - no-floor-and-ceiling points
    - *heightmap
    - figure containing all plots
"""

class SaveTask:

    def __init__(self, num_workers: int = 0) -> None:
        """
        Initialize a SaveTask instance.

        :param num_workers: The number of worker threads to use for processing save jobs. If 0, it will be set to the number of CPU cores minus one.
        :type num_workers: int
        """
        # Create a queue that we will use to store our "workload".
        self.queue = asyncio.Queue()
        self.deferred_jobs = []

        # Use a dedicated executor so save jobs do not compete with model work
        # running in the default asyncio threadpool.
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers if num_workers > 0 else self.recommend_workers(),
            thread_name_prefix="save-worker",
        )

        # Create num_workers worker tasks to process the queue concurrently.
        self.workers = [asyncio.create_task(self._worker()) for _ in range(num_workers)]
        
        # Misc.: Set up name of overall task and logger
        self.logger = logging.getLogger(__name__)

    # -----------------------------
    # Special Methods
    # -----------------------------   

    def __call__(self, save_func, *args, **kwargs) -> None:
        """
        Puts a "work item" into the queue,
        which consists of the save function and its arguments. 

        :param save_func: The function to be called for saving.
        :type save_func: Callable
        :param args: Positional arguments for the save function.
        :param kwargs: Keyword arguments for the save function.
        """
        self.queue.put_nowait((save_func, args, kwargs))

    def __len__(self) -> int:
        """
        Returns the number of "work items" currently in the queue.
        """
        return self.queue.qsize()

    def stage(self, save_func, *args, **kwargs) -> None:
        """Buffer a save job to be flushed later."""
        self.deferred_jobs.append((save_func, args, kwargs))

    def flush_staged(self) -> None:
        """Submit all staged save jobs to the worker queue."""
        for save_func, args, kwargs in self.deferred_jobs:
            self(save_func, *args, **kwargs)
        self.logger.info(f"Queued {len(self.deferred_jobs)} deferred save jobs")
        self.deferred_jobs.clear()

    def clear_staged(self) -> None:
        """Drop staged jobs without queueing them."""
        self.deferred_jobs.clear()

    def shutdown(self) -> None:
        """Shuts down the SaveTask by canceling all worker tasks and shutting down the executor."""
        for worker in self.workers:
            worker.cancel()
        self.executor.shutdown(wait=True)

    def is_done(self) -> bool:
        """Checks if all worker tasks have completed."""
        return all(worker.done() for worker in self.workers)
    
    def recommend_workers(self) -> int:
        """Recommends a number of worker threads based on the number of CPU cores."""
        cpu_count = os.cpu_count() or 1
        return max(1, cpu_count - 1)

    # -----------------------------
    # Private Methods
    # -----------------------------       

    async def _worker(self):
        """
        Continuously processes all queued tasks
        """
        while True:
            # Get a "work item" out of the queue.
            save_func, args, kwargs = await self.queue.get()

            try:
                loop = asyncio.get_running_loop()
                job = partial(save_func, *args, **kwargs)
                await loop.run_in_executor(self.executor, job)
            except Exception as e:
                self.logger.error(f"Error saving file: {e}")
            finally:                
                # Notify the queue that the "work item" has been processed.
                self.queue.task_done()

# -----------------------------
# Utility
# -----------------------------