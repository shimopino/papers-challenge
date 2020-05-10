import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config.yml")
def my_app(cfg: DictConfig) -> None:
    print(cfg.pretty())
    print("---")
    print(cfg)

    
if __name__ == "__main__":
    my_app()