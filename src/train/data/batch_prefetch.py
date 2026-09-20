"""Bounded CPU batch production; the producer alone owns and closes its source."""
from contextlib import closing
from queue import Empty, Full, Queue
from threading import Event, Thread


def prefetched_batches(source_factory, capacity):
    if capacity < 1:
        raise ValueError("Batch prefetch capacity must be positive")
    queue = Queue(maxsize=capacity)
    cancelled = Event()

    def offer(kind, value):
        while not cancelled.is_set():
            try:
                queue.put((kind, value), timeout=0.05)
                return True
            except Full:
                pass
        return False

    def produce():
        try:
            with closing(source_factory(cancelled)) as source:
                while not cancelled.is_set():
                    try:
                        batch = next(source)
                    except StopIteration:
                        break
                    if not offer("batch", batch):
                        return
                    del batch
            offer("done", None)
        except BaseException as error:
            offer("error", error)

    worker = Thread(target=produce, name="batch-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            kind, value = queue.get()
            if kind == "done":
                return
            if kind == "error":
                raise value
            yield value
            del value
    finally:
        cancelled.set()
        worker.join()
        # Release queued tensors even when the consumer stopped early.
        while True:
            try:
                queue.get_nowait()
            except Empty:
                break
