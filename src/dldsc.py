from train.train import train_dldsc
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=".", config_name="config")
def train(cfg: DictConfig):
    train_dldsc(cfg)

if __name__ == '__main__':
    train()