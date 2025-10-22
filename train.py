from logging import getLogger
from pathlib import Path
from pprint import pprint
from sys import path


import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything

# set seeds for sklearn
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader

from src.utils.data import get_dataloder

logger = getLogger(__name__)


@hydra.main(
    config_path="./configs/train",
    config_name="example_contrastive", # TODO: change config name
    version_base="1.3",
)
def main(cfg: DictConfig):
    # TODO: set the device!
    # TODO: missing loss

    ##############################
    ####### DATASET SETUP ########
    ##############################
    dataset = instantiate(cfg.dataset)
    # TODO: add other split possibilitiesn using custom classes
    train_set, val_set, test_set = dataset.get_splits()

    train_loader = DataLoader(
        train_set,
        **cfg.dataloader,
    )
    val_loader = DataLoader(
        val_set,
        **cfg.dataloader,
    )
    test_loader = DataLoader(
        test_set,
        **cfg.dataloader,
    )

    #############################
    ####### MODELS SETUP ########
    #############################
    engine = instantiate(cfg.engine)

    trainer = instantiate(cfg.trainer)

    #############################
    ####### MODELS TRAIN ########
    #############################

    trainer.fit(engine, train_loader, val_loader)

    trainer.test(engine, test_loader, ckpt_path="best")


if __name__ == "__main__":
    main()
