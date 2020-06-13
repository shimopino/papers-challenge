import math
import random
import torch
import torch.nn.functional as F


class RandomResizedCropTensor:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), p=0.5):

        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.p = p

    def apply(self, single_tensor):
        r"""Radndomly Resize and Crop torch.Tensor
        :param single_tensor: torch.Tensor [C, H, W]
        """

        H, W = single_tensor.shape[1], single_tensor.shape[2]
        area = H * W

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = math.log(self.ratio[0]), math.log(self.ratio[1])
            aspect_ratio = math.exp(random.uniform(*log_ratio))
            z = None
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if (0 < w <= W) and (0 < h <= H):
                i = random.randint(0, H - h)
                j = random.randint(0, W - w)
                z = i, j, h, w
                break

        if z is None:
            in_ratio = float(W) / float(H)
            if (in_ratio < min(self.ratio)):
                w = W
                h = int(round(w / min(self.ratio)))
            elif (in_ratio > max(self.ratio)):
                h = H
                w = int(round(h * max(self.ratio)))
            else:
                w = W
                h = H

            i = (H - h) // 2
            j = (W - w) // 2
            z = i, j, h, w

        return F.interpolate(single_tensor[:, i:i + h, j:j + w].unsqueeze(0),
                             size=self.size,
                             mode="bilinear")[0]

    def __call__(self, image_tensor):
        r"""apply RandomResizedCrop to 4D-Tensor
        :param image_tensor: torch.Tensor [B, C, H, W]
        """

        return torch.stack([self.apply(image) if random.random() > self.p else image
                            for image in image_tensor], dim=0)


class RandomHorizontalFlipTensor:
    def __init__(self, p=0.5):
        self.p = p

    def apply(self, single_tensor):
        r"""Radndomly Horizontally Flip torch.Tensor
        :param single_tensor: torch.Tensor [C, H, W]
        """

        return torch.flip(single_tensor, dims=(2, ))

    def __call__(self, image_tensor):
        r"""apply Random Horizontally Flipping to 4D-tensor
        :param image_tensor: torch.Tensor [B, C, H, W]
        """

        return torch.stack([self.apply(image) if random.random() > self.p else image
                            for image in image_tensor])


if __name__ == "__main__":

    sample = torch.rand(10, 3, 64, 64).to("cuda:0")
    sample.mul_(2).sub_(1)

    H, W = sample.shape[2], sample.shape[3]
    size = (H, W)

    tranform_rcrop = RandomResizedCropTensor(size)
    sample_cropped = tranform_rcrop(sample)

    transform_flip = RandomHorizontalFlipTensor()
    sample_flip = transform_flip(sample)
