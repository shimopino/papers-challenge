import time
import logging
from typing import Optional
from contextlib import contextmanager


@contextmanager
def timer(name: str, logger: Optional[logging.logger] = None):
    start = time.time()
    yield
    message = f"[{name}] done in {time.time() - start}"
    if logger:
        logger.info(message)
    else:
        print(message)
