import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

files = glob.glob("samples/*")
imgs = np.array([mpimg.imread(file) for file in files[:8]])

def get_lambda(shape, beta, same_on_batch=True):
    if same_on_batch:
        lam = np.random.beta(beta, beta)
    else:
        lam = np.random.beta(beta, beta, size=(shape[0],))
    return lam

def rand_bbox(shape, lam, same_on_batch=True):
    B, W, H, C = shape
    cutmix_ratio = np.sqrt(1. - lam)

    cut_w = (W * cutmix_ratio).astype(np.int)
    cut_h = (H * cutmix_ratio).astype(np.int)

    if same_on_batch:
        cx = np.random.randint(W)
        cy = np.random.randint(H)
    else:
        cx = np.random.randint(W, size=(B,))
        cy = np.random.randint(H, size=(B,))

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def create_cutmix_mask(shape, bbx1, bby1, bbx2, bby2, same_on_batch=True):

    if same_on_batch:
        bbx1 = np.atleast_1d(bbx1)
        bby1 = np.atleast_1d(bby1)
        bbx2 = np.atleast_1d(bbx2)
        bby2 = np.atleast_1d(bby2)

    B, W, H, _ = shape

    masks = np.ones(shape=(B, W, H, 1), dtype=np.int)
    for i, bbox in enumerate(zip(bbx1, bby1, bbx2, bby2)):
        x1, y1, x2, y2 = bbox
        if same_on_batch:
            masks[:, x1:x2, y1:y2, :] = 0
            break
        else:
            masks[i, x1:x2, y1:y2, :] = 0

    return masks

def apply_cutmix(tensor, masks, same_on_batch=True):

    rand_idx = np.arange(len(tensor))
    np.random.shuffle(rand_idx)
    mix_tensor = tensor[rand_idx].copy()

    if same_on_batch:
        apply_mask = masks[0, None, ...]
        cutmix_tensor = tensor * apply_mask + (1 - apply_mask) * mix_tensor
    else:
        cutmix_tensor = [tensor[i] * mask + mix_tensor[i] * (1 - mask) \
                         for i, mask in enumerate(masks)]
        cutmix_tensor = np.stack(cutmix_tensor, axis=0)

    return cutmix_tensor
