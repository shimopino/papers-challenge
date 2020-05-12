import os
import hydra
from SinGAN.manupulate import SinGAN_generate
from SinGAN.training import train
from SinGAN.functions import read_image, adjust_scales2image, generate_dir2save


@hydra.main(config_path="config.yml")
def main(cfg):
    print("... configuration ...")
    print(cfg.pretty())

    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = generate_dir2save(cfg)

    if os.path.exists(dir2save):
        print("trained model already exist")
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = read_image(cfg)
        adjust_scales2image(real, cfg)
        train(cfg, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs, Zs, reals, NoiseAmp, cfg)


if __name__ == "__main__":
    main()
