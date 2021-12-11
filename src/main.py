import hydra, yaml, os
from omegaconf import OmegaConf, DictConfig, open_dict
from engine import main

# args = OmegaConf.load('/nfs/Workspace/brats_brain_segmentation/src/config/config.yaml')

#     main(cfg, 0)

# CONFIG_PATH = "/nfs/Workspace/brats_brain_segmentation/src/"

# def load_config(config_name):
#     with open(os.path.join(CONFIG_PATH, config_name)) as f:
#         conf = yaml.safe_load(f)

#     return conf

# args = load_config("config.yaml")


@hydra.main(config_path="config", config_name="config")
def func(cfg: DictConfig):
    main(cfg, 0)


if __name__ == "__main__":
    func()
