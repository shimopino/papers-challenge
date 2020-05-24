from pathlib import Path
from logging import getLogger, StreamHandler, FileHandler, Formatter, INFO
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


class Logger:
    def __init__(self, log_dir, flush_secs):

        self.log_dir = Path(log_dir)
        self.flush_secs = flush_secs
        self.logger = self._build_logger()
        self.writers = {}

    def _build_logger(self):

        # check log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger = getLogger("sample")
        logger.setLevel(INFO)
        # format
        fmt_str = "[%(levelname)s] %(asctime)s: \t%(message)s"
        fmt = Formatter(fmt_str, "%Y-%m-%d %H-%M-%S")
        # stream handler
        stm_handler = StreamHandler()
        stm_handler.setLevel(INFO)
        stm_handler.setFormatter(fmt)
        # file handler
        file_handler = FileHandler(self.log_dir / "logs.txt", mode="w")
        file_handler.setLevel(INFO)
        file_handler.setFormatter(fmt)
        # add handler to logger
        logger.addHandler(stm_handler)
        logger.addHandler(file_handler)

        return logger

    def print_log(self, message):
        self.logger.info(message)

    def _build_writer(self, metrics):
        writer = SummaryWriter(
            log_dir=self.log_dir / "data" / metrics, flush_secs=self.flush_secs
        )
        return writer

    def add_scalar(self, metrics, value, global_step=None):
        if metrics not in self.writers:
            self.writers[metrics] = self._build_writer(metrics)

        self.writers[metrics].add_scalar(metrics, value, global_step)

    def add_image(self, tag, tensor, global_step=None):
        n = min(tensor.size(0), 36)
        image_tensor = make_grid(tensor[:n], nrow=6, padding=2, normalize=True)

        if tag not in self.writers:
            self.writers[tag] = self._build_writer(tag)

        self.writers[tag].add_image(tag, image_tensor, global_step)

    def add_histogram(self, tag, tensor, global_step=None):

        if tag not in self.writers:
            self.writers[tag] = self._build_writer(tag)

        self.writers[tag].add_histogram(tag, tensor, global_step)

    def add_embedding(self, tag, emb_mat, labels=None, global_step=None):
        mat = emb_mat.transpose(0, 1)

        if tag not in self.writers:
            self.writers[tag] = self._build_writer(tag)

        self.writers[tag].add_embedding(mat, metadata=labels, global_step=global_step)

    def close_writers(self):
        for metrics in self.writers:
            self.writers[metrics].close()


if __name__ == "__main__":
    import shutil
    import torch

    def setup(cfg):
        logger = Logger(cfg["log_dir"], cfg["flush_secs"])
        return logger

    cfg = {"log_dir": "./logs", "flush_secs": 1}
    logger = setup(cfg)

    global_iters = 1

    # test logger functionality
    logger.print_log(f"sample output to stdout: {global_iters}")
    # test scalar metrics output
    logger.add_scalar("sample", 1.0, global_step=global_iters)
    # test tensor image [b, 3, H, W] output
    logger.add_image("sample_img", torch.randn(10, 3, 64, 64), global_step=global_iters)
    # test weight histogram
    conv = torch.nn.Conv2d(32, 64, 3, 1, 1, bias=False)
    logger.add_histogram("conv sample", conv.weight.data, global_step=global_iters)
    # test embedding output
    embed = torch.nn.Embedding(10, 128)
    logger.add_embedding(
        "sample embedding", embed.weight.data, global_step=global_iters
    )

    def teardown(logger):
        logger.close_writers()

        path = Path(logger.log_dir)
        shutil.rmtree(path)

    teardown(logger)
