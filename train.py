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
    config_path="./configs/classification",
    # config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):

    ##############################
    ####### DATASET SETUP ########
    ##############################
    dataset = instantiate(cfg.dataset)
    train_set, val_set, test_set = dataset.get_splits()

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
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
