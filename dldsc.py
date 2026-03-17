from src.train.train import train
from src.inference.inference import predict
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    if cfg.mode == 'Train':
        train(cfg)
    elif cfg.mode == 'Inference':
        predict(cfg)
    else:
        raise ValueError(f"{cfg.mode} is an invalid mode.")

if __name__ == '__main__':
    main()