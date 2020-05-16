# https://github.com/kwotsin/mimicry/blob/22497cb3738214b212cd2c2a7b0867e7836b1f82/torch_mimicry/training/logger.py
import numpy as np
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from base_logger import create_logger, get_logger


class Logger:
    """
    take log on both Tensorboard and logging

    Arguments:
        log_dir {[str]} -- root directory for storing logs
        experiment {[str]} -- directory name for logging
        num_steps {[int]} -- #of examples in the dataset
        dataset_size {[int]} -- [description]
        device {[str]} -- torch.Tensor device to data

    Keyword Arguments:
        flush_secs {int} -- waiting seconds for flushing log (default: {120})
    """

    def __init__(
        self,
        log_dir,
        experiment,
        num_steps,
        dataset_size,
        device,
        flush_secs=120,
        **kwargs,
    ):
        self.log_dir = log_dir
        self.experiment = experiment
        self.num_steps = num_steps
        self.dataset_size = dataset_size
        self.flush_secs = flush_secs
        self.device = device
        self.num_epochs = self._get_epoch(num_steps)
        self.writers = {}
        self.logger = self._get_logger()

    def _get_logger(self):
        """
        get logger for File/Stream handlers
        """
        create_logger(self.log_dir, self.experiment)
        return get_logger()

    def _get_epoch(self, num_steps):
        """
        Arguments:
            num_steps {[int]} -- #of total iterations

        Returns:
            [int] -- #of epochs
        """
        return max(int(num_steps / self.dataset_size), 1)

    def _build_writer(self, metrics):
        """
        Arguments:
            metrics {[str]} -- folder path where you output log

        Returns:
            [writer] -- Tensorboard Writer Instance
        """
        writer = SummaryWriter(
            log_dir=Path(self.log_dir).joinpath("data", metrics),
            flush_secs=self.flush_secs,
        )
        return writer

    def write_summaries(self, log_data, global_step):
        """
        add dictionary-based metrics to tensorboard

        Arguments:
            log_data {[dict]} -- dict type metrics like {metric name: value}
            global_step {[int]} -- iteraion step
        """
        for metrics, data in log_data.items():
            if metrics not in self.writers:
                self.writers[metrics] = self._build_writer(metrics)

            name = log_data.get_group_name(metrics) or metrics
            self.writers[metrics].add_scalar(name, log_data[metrics], global_step)

    def close_writers(self):
        """
        close all writers
        """
        for metrics in self.writers:
            self.writers[metrics].close()

    def print_log(self, global_step, log_data, time_taken):
        """
        add model metrics to logger

        Arguments:
            global_step {[int]} -- iteraion step
            log_data {[dict]} -- dict-based log data
            time_taken {[float]} -- Time spending for a specific execution
        """

        basic_log = [
            "[Epoch {:d}/{:d}][Global Step: {:d}/{:d}]".format(
                self._get_epoch(global_step),
                self.num_epochs,
                global_step,
                self.num_steps,
            )
        ]

        # get all metric name, and then sort them
        additional_logs = [""]
        metric_names = sorted(log_data.keys())

        # add all metric value
        for metric_name in metric_names:
            additional_logs.append("{}: {}".format(metric_name, log_data[metric_name]))

        # add execution time
        additional_logs.append("({:.4f} sec/idx)".format(time_taken))

        # Accumulate to log
        basic_log.append("\n| ".join(additional_logs))

        # Finally print the output
        ret = " ".join(basic_log)
        self.logger.info(ret)

    def _get_fixed_noise(self, nz, num_images, output_dir=None):
        """
        get fixed noize vector for reproducibility

        Arguments:
            nz {[int]} -- #of dimensions for latent code
            num_images {[int]} -- #of images for output

        Keyword Arguments:
            output_dir {[str]} -- outpu directory for generated images (default: {None})

        Returns:
            [torch.Tensor] -- torch.Tensor [num_images, nz]
        """

        if output_dir is None:
            output_dir = Path(self.log_dir).joinpath("viz")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir.joinpath("fixed_noise_nz_{}.pth".format(nz))

        if output_file.exists():
            noise = torch.load(output_file)

        else:
            noise = torch.randn((num_images, nz))
            torch.save(noise, output_file)

        return noise.to(self.device)

    def _get_fixed_labels(self, num_images, num_classes):
        """
        return fixed label tensor

        Arguments:
            num_images {[int]} -- #of images
            num_classes {[int]} -- #of classes

        Returns:
            [torch.Tensor] -- labels tensor like [0, 1, 2, 0, 1, 2, 0, ...]
        """
        labels = np.array([i % num_classes for i in range(num_images)])
        labels = torch.from_numpy(labels).to(self.device)

        return labels

    def vis_images(self, netG, global_step, num_images=64):
        """
        Produce visualisations of the G(z), one fixed and one random.

        Arguments:
            netG {[Module]}: Generator model for producing images.
            global_step {[int]}: iteration steps.
            num_images {[int]}: #of images to visualise.
        """

        img_dir = Path(self.log).joinpath("images")
        img_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            # Generate random images
            noise = torch.randn((num_images, netG.nz), device=self.device)
            fake_images = netG(noise).detach().cpu()

            # Generate fixed random images
            fixed_noise = self._get_fixed_noise(nz=netG.nz, num_images=num_images)

            if hasattr(netG, "num_classes") and netG.num_classes > 0:
                fixed_labels = self._get_fixed_labels(num_images, netG.num_classes)
                fixed_fake_images = netG(fixed_noise, fixed_labels).detach().cpu()
            else:
                fixed_fake_images = netG(fixed_noise).detach().cpu()

            # Map name to results
            images_dict = {"fixed_fake": fixed_fake_images, "fake": fake_images}

            # Visualise all results
            for name, images in images_dict.items():
                images_viz = vutils.make_grid(images, padding=2, normalize=True)

                vutils.save_image(
                    images_viz,
                    "{}/{}_samples_step_{}.png".format(img_dir, name, global_step),
                    normalize=True,
                )

                if "img" not in self.writers:
                    self.writers["img"] = self._build_writer("img")

                self.writers["img"].add_image(
                    "{}_vis".format(name), images_viz, global_step=global_step
                )
