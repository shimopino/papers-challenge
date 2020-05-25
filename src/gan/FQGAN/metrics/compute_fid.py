import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fid_utils import get_random_images, get_inception_model, _normalize_images



def get_activations(images, model, batch_size=50, dims=2048, device="cpu"):
    """
    Calculates the activations of the pool_3 layer for all images.

    Parameters: 
        - images: ndarray of [B, W, H, C] in range [0, 255]
    """
    model = model.to(device)
    model.eval()
    pred_arr = np.empty((len(images), dims))

    for i in tqdm(range(0, len(images), batch_size)):
        start = i
        end = i + batch_size

        images_batch = images[start:end]
        # PIL Image [B, W, H, C] --> Tensor [B, C, H, W]
        images_batch = images_batch.transpose(0, 3, 2, 1)
        images_batch /= 255

        batch = torch.from_numpy(images_batch).type(torch.FloatTensor)
        batch = batch.to(device)

        pred = model(batch)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().numpy().reshape(pred.size(0), -1)

    return pred_arr


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from datasets import data_utils

    dataset = data_utils.get_celeba_dataset("../data/celeba", 128)
    print(dataset[0][0].size())
    print(dataset.imgs[0][0])

    # sample img shape: [178, 218] == [W, H]
    sample_img = Image.open(dataset.imgs[0][0])
    print(sample_img.size)

    # get scoring images: [B, W, H, C]
    images = get_random_images(dataset, 1000)
    print(images.shape)

    # get Inception Model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    print(block_idx)

    # get activations
    preds_real = get_activations(images, model, device="cuda:0")
    print(preds_real.shape)
