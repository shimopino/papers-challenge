import numpy as np
from scipy import linalg
import torch
from inception_model import InceptionV3


def get_inception_model(dims, device):

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model = model.to(device)

    return model


def get_random_images(dataset, num_samples):

    choices = np.random.choice(range(len(dataset)), size=num_samples, replace=False)

    images = []
    for choice in choices:
        img = np.array(dataset[choice][0])
        img = np.expand_dims(img, axis=0)
        images.append(img)
    images = np.concatenate(images, axis=0)

    return images


def _normalize_images(images):
    """
    Given a tensor of images, uses the torchvision
    normalization method to convert floating point data to integers. See reference
    at: https://pytorch.org/docs/stable/_modules/torchvision/utils.html#save_image
    The function uses the normalization from make_grid and save_image functions.
    Args:
        images (Tensor): Batch of images of shape (N, 3, H, W).
    Returns:
        ndarray: Batch of normalized images of shape (N, H, W, 3).
    """
    # Shift the image from [-1, 1] range to [0, 1] range.
    min_val = float(images.min())
    max_val = float(images.max())
    images.clamp_(min=min_val, max=max_val)
    images.add_(-min_val).div_(max_val - min_val + 1e-5)

    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    images = (
        images.mul_(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(0, 2, 3, 1)
        .to("cpu", torch.uint8)
        .numpy()
    )

    return images


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    if mu1.shape != mu2.shape or sigma1.shape != sigma2.shape:
        raise ValueError(
            "(mu1, sigma1) should have exactly the same shape as (mu2, sigma2)."
        )

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            "WARNING: fid calculation produces singular product; adding {} to diagonal of cov estimates".format(
                eps
            )
        )

        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(images, sess, batch_size=50, verbose=True):
    """
    Calculation of the statistics used by the FID.
    Args:
        images (ndarray): Numpy array of shape (N, H, W, 3) and values in
            the range [0, 255].
        sess (Session): TensorFlow session object.
        batch_size (int): Batch size for inference.
        verbose (bool): If True, prints out logging information.
    Returns:
        ndarray: Mean of inception features from samples.
        ndarray: Covariance of inception features from samples.
    """
    act = inception_utils.get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma


if __name__ == "__main__":
    images = torch.randn(10, 3, 64, 64)
    print(images.shape)
    print("images: min {}, max {}".format(images.min(), images.max()))

    images_normalized = _normalize_images(images)
    print(images_normalized.shape)
    print(
        "images_normalized: min {}, max {}".format(
            images_normalized.min(), images_normalized.max()
        )
    )
