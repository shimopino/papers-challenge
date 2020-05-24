from torchvision import transforms
from torchvision import datasets as dsets


def get_celeba_dataset(root, image_size, transform_data=True):

    transforms_list = []
    if transform_data:
        transforms_list += [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    data_transforms = transforms.Compose(transforms_list)
    dataset = dsets.ImageFolder(root=root, transform=data_transforms)

    return dataset


if __name__ == "__main__":
    dataset = get_celeba_dataset("../../../data", 128)

    print(len(dataset))
