from datetime import datetime
from pathlib import Path
from logging import DEBUG
from logging import getLogger, StreamHandler, FileHandler, Formatter


def create_logger(log_dir, experiment):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = Path(log_dir).joinpath(experiment, timestamp)

    # logger
    logger = getLogger(experiment, mode="w")
    logger.setLevel(DEBUG)

    # formatter
    fmt = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    file_handler = FileHandler(log_file)
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(fmt)

    # stream_handler
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(fmt)

    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def get_logger(experiment):
    return getLogger(experiment)
