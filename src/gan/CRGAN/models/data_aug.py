import numpy as np
from kornia.filters import GaussianBlur2d
from kornia.augmentation import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomGrayscale
)


class KorniaCompose:

    def __init__(self, tensor_transforms):
        self.tensor_transforms = tensor_transforms

    def __call__(self, img_tensor):

        for tensor_t in self.tensor_transforms:
            img_tensor = tensor_t(img_tensor)

        return img_tensor


class SimCLRAugmentation:

    def __init__(self, input_shape, s=1.0, apply_transforms=None):

        assert len(input_shape) == 3, "input_shape should be (H, W, C)"

        self.input_shape = input_shape
        self.H, self.W, self.C = input_shape[0], input_shape[1], input_shape[2]
        self.s = s
        self.apply_transforms = apply_transforms

        if self.apply_transforms is None:
            kernel_size = int(0.1 * self.H)
            sigma = self._get_sigma()

            self.apply_transforms = KorniaCompose([
                RandomResizedCrop(size=(self.H, self.W), scale=(0.08, 1.0)),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s),
                RandomGrayscale(p=0.2),
                GaussianBlur2d(kernel_size=(kernel_size, kernel_size),
                               sigma=(sigma, sigma))
            ])

    def _get_sigma(self):

        min = 0.1
        max = 2.0
        sigma = (max - min) * np.random.random_sample() + min

        return sigma

    def __call__(self, img_tensor):

        return self.apply_transforms(img_tensor)


if __name__ == "__main__":
    import torch

    input_shape = (256, 256, 3)
    img = torch.randn(10, 3, 256, 256)

    print(img.shape)
    print(SimCLRAugmentation(input_shape)(img).shape)