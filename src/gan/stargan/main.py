import os
from config import get_args
from trainer import Trainer
from datasets import get_loader
# from torch.backends import cudnn


def main(config):
    # For fast training.
    # cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)

    # Trainer for training and testing StarGAN.
    trainer = Trainer(celeba_loader, rafd_loader, config)

    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            trainer.train()
        elif config.dataset in ['Both']:
            trainer.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            trainer.test()
        elif config.dataset in ['Both']:
            trainer.test_multi()


if __name__ == "__main__":
    config = get_args()
    print(config)
    main(config)
