# https://github.com/kwotsin/mimicry/blob/master/torch_mimicry/modules/losses.py
import torch
import torch.nn.functional as F


def minmax_loss_dis(
    output_fake, output_real, fake_label_value=0.0, real_label_value=1.0, **kwargs
):

    fake_labels = torch.full(
        (output_fake.shape[0], 1), fake_label_value, device=output_fake.device
    )
    real_labels = torch.full(
        (output_real.shape[0], 1), real_label_value, device=output_real.device
    )

    errD_fake = F.binary_cross_entropy_with_logits(output_fake, fake_labels, **kwargs)
    errD_real = F.binary_cross_entropy_with_logits(output_real, real_labels, **kwargs)

    loss = errD_fake + errD_real
    return loss


def minmax_loss_gen(output_fake, real_label_value=1.0, **kwargs):

    real_labels = torch.full(
        (output_fake.shape[0], 1), real_label_value, device=output_fake.device
    )

    loss = F.binary_cross_entropy_with_logits(output_fake, real_labels, **kwargs)
    return loss


def wasserstein_loss_dis(output_fake, output_real):

    loss = -1.0 * output_real.mean() + output_fake.mean()
    return loss


def wasserstein_loss_gen(output_fake):

    loss = -output_fake.mean()
    return loss


def hinge_loss_dis(output_fake, output_real):

    loss = F.relu(1.0 - output_real).mean() + F.relu(1.0 + output_fake).mean()
    return loss


def hinge_loss_gen(output_fake):

    loss = -output_fake.mean()
    return loss


if __name__ == "__main__":
    from logging import DEBUG
    from logging import getLogger, StreamHandler, Formatter

    # logger
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    # stream_handler
    fmt_str = "[%(levelname)s] %(asctime)s >>\t%(message)s"
    format = Formatter(fmt_str, "%Y-%m-%d %H:%M:%S")
    stream_handler = StreamHandler()
    stream_handler.setFormatter(format)
    stream_handler.setLevel(DEBUG)
    # set handler
    logger.addHandler(stream_handler)

    # sample discriminator output probability
    output_fake = torch.ones(4, 1)
    output_real = torch.ones(4, 1)
    logger.debug(f"output fake with ones: {output_fake.size()}")
    logger.debug(f"output real with ones: {output_real.size()}")

    # minmax
    logger.debug("--------------------------------")
    loss_gen = minmax_loss_gen(output_fake)
    loss_dis = minmax_loss_dis(output_fake, output_real)
    logger.debug(f"generator loss    : {loss_gen.item()}")
    logger.debug(f"discriminator loss: {loss_dis.item()}")

    # wasserstein
    logger.debug("--------------------------------")
    loss_gen = wasserstein_loss_gen(output_fake)
    loss_dis = wasserstein_loss_dis(output_fake, output_real)
    logger.debug(f"generator loss    : {loss_gen.item()}")
    logger.debug(f"discriminator loss: {loss_dis.item()}")

    # hinge
    logger.debug("--------------------------------")
    loss_gen = hinge_loss_gen(output_fake)
    loss_dis = hinge_loss_dis(output_fake, output_real)
    logger.debug(f"generator loss    : {loss_gen.item()}")
    logger.debug(f"discriminator loss: {loss_dis.item()}")
