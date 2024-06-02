import logging

from contextlib import contextmanager


@contextmanager
def disable_pymc_sampling_logs(logger: logging.Logger = logging.getLogger("pymc")):
    effective_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(effective_level)
