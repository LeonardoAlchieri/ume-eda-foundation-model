from logging import getLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from torch.utils.data import DataLoader

from edamame.utils.logging import log_hydra_config_to_lightning

logger = getLogger(__name__)


@hydra.main(
    config_path="./configs/train",
    config_name="reconstruction_sweep",  # TODO: change config name
    version_base="1.3",
)
def main(cfg: DictConfig):

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
    engine = instantiate(
        cfg.engine, scaling_params=instantiate(cfg.dataset_scaling_fun)(dataset=train_set)
    )

    trainer = instantiate(cfg.trainer)

    #############################
    ### LOG HYDRA CONFIG ########
    #############################
    log_hydra_config_to_lightning(trainer, cfg)

    #############################
    ####### MODELS TRAIN ########
    #############################

    trainer.fit(engine, train_loader, val_loader)

    # trainer.test(engine, test_loader, ckpt_path="best")
    trainer.test(engine, test_loader)


if __name__ == "__main__":
    main()
