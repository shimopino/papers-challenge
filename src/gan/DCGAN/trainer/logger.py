from datetime import datetime
from pathlib import Path
from logging import getLogger, Formatter, INFO, FileHandler, StreamHandler
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, flush_secs):

        self.log_dir = Path(log_dir)
        self.flush_secs = flush_secs
        self.logger = self._build_logger()
        self.timestamp = self._get_timestamp()
        self.writers = {}

    def _get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%H")

    def _build_logger(self):
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        # format
        fmt_str = "[%(levelname)s] %(asctime)s: \t%(message)s"
        fmt = logging.Formatter(fmt_str, "%Y-%m-%d %H-%M-%S")
        # stream handler
        stm_handler = logging.StreamHandler(sys.stdout)
        stm_handler.setLevel(logging.INFO)
        stm_handler.setFormatter(fmt)
        # file handler
        path = self.log_dir.joinpath(self.timestamp, "logs.txt")
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)
        # add handler to logger
        logger.addHandler(stm_handler)
        logger.addHandler(file_handler)

        return logger

    def print_log(self, message):
        self.logger.info(message)

    def _build_writer(self, metrics):
        writer = SummaryWriter(
            log_dir=self.log_dir.joinpath(self.timestamp, metrics),
            flush_secs=self.flush_secs
        )
        return writer

    def add_scalar_log(self, metrics, value, global_step=None):
        if metrics not in self.writers:
            self.writers[metrics] = self._build_writer(metrics)

        self.writers[metrics].add_scalar(metrics, value, global_step)

    def add_images(self, tag, tensor, global_step=None):
        n = min(tensor.size(0), 36)
        image_tensor = make_grid(tensor[:n], nrow=6, padding=2, normalize=True)

        if tag not in self.writers:
            self.writers[tag] = self._build_writer(tag)

        self.writers[tag].add_image(tag, image_tensor, global_step)

    def add_histogram(self, tag, value, global_step=None):

        if tag not in self.writers:
            self.writers[tag] = self._build_writer(tag)

        self.writers[tag].add_histogram(tag, value, global_step)

    def close_writers(self):
        for metrics in self.writers:
            self.writers[metrics].close()