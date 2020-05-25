import numpy as np
import torch
import tensorflow as tf
from fid_utils import calculate_activation_statistics, calculate_frechet_distance


def get_random_real_images(dataset, num_samples):
    """return ndarray [B, H, W, 3] to range [0, 255]"""

    choices = np.random.choice(range(len(dataset)), size=num_samples, replace=True)

    images = []
    for choice in choices:
        images.append(dataset[choice][0].unsqueeze(0))

    images = torch.cat(images, 0)
    images = _normalize_images(images)

    return images


def get_random_generated_images(netG, num_samples, batch_size, device):

    images = []
    netG.eval()
    with torch.no_grad():
        for idx in range(num_samples // batch_size):
            batch = netG.generate_images(batch_size, device).detach().cpu()
            images.append(batch)

        images = torch.cat(images, 0)  # ndarray [B, C, H, W] to range [-1, 1]
        images = _normalize_images(images)  # tensors [B, H, W, C] to range [0, 255]

    return images


def _normalize_images(images):

    # shift the image from [-1, 1] range to [0, 1] range
    min_val = float(images.min())
    max_val = float(images.max())
    images.clamp_(min=min_val, max=max_val)
    images.add_(-min_val).div_(max_val - min_val + 1e-5)

    # add 0.5 after unnormalizing to [0, 255] to round to nearest
    # transpose [B, 3, H, W] --> [B, H, W, 3]
    images = (
        images.mul_(255)
        .add_(0.5)
        .clamp(0, 255)
        .permute(0, 2, 3, 1)
        .to("cpu", dtype=torch.uint8)
        .numpy()
    )

    return images


def compute_real_dist_stats(
    dataset, num_samples, batch_size, sess, verbose=True, log_dir=None, seed=None
):
    real_images = get_random_real_images(dataset, num_samples)
    m_real, s_real = calculate_activation_statistics(
        real_images, sess, batch_size, verbose
    )
    return m_real, s_real


def compute_fake_dist_stats(
    netG, num_samples, batch_size, sess, device, verbose=True, seed=None
):
    fake_images = get_random_generated_images(netG, num_samples, batch_size, device)
    m_fake, s_fake = calculate_activation_statistics(
        fake_images, sess, batch_size, verbose
    )
    return m_fake, s_fake


def get_fid(
    dataset,
    netG,
    num_samples,
    batch_size,
    device,
    verbose=True,
    log_dir=None,
    seed=None,
):
    if device and device.index is not None:
        # Avoid unbounded memory usage
        gpu_options = tf.GPUOptions(
            allow_growth=True,
            per_process_gpu_memory_fraction=0.15,
            visible_device_list=str(device.index),
        )
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    else:
        config = tf.compat.v1.ConfigProto(device_count={"GPU": 0})

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        m_real, s_real = compute_real_dist_stats(dataset, num_samples, batch_size, sess)

        m_fake, s_fake = compute_fake_dist_stats(
            netG, num_samples, batch_size, sess, device
        )

        FID_score = calculate_frechet_distance(
            mu1=m_real, sigma1=s_real, mu2=m_fake, sigma2=s_fake
        )

    return FID_score
