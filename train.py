from logging import getLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader

from edamame.engine import BaseSelfSupervisedEngine
from edamame.pipeline.edamame_pipeline import EdamamePipeline
from edamame.utils.logging import log_hydra_config_to_lightning

logger = getLogger(__name__)


def pre_train(cfg: DictConfig) -> tuple[BaseSelfSupervisedEngine, Logger]:
    ##############################
    ####### DATASET SETUP ########
    ##############################
    dataset = instantiate(cfg.dataset)
    # TODO: add other split possibilities using custom classes
    train_set, val_set, test_set = dataset.get_splits()

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **cfg.dataloader,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        **cfg.dataloader,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **cfg.dataloader,
    )

    #############################
    ####### MODELS SETUP ########
    #############################
    engine = instantiate(
        cfg.engine,
        scaling_params=instantiate(cfg.dataset_scaling_fun)(dataset=train_set),
    )

    trainer = instantiate(cfg.trainer)

    #############################
    ### LOG HYDRA CONFIG ########
    #############################
    log_hydra_config_to_lightning(trainer, cfg)

    #############################
    ####### MODELS TRAIN ########
    #############################
    try:
        trainer.fit(engine, train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        print("NO TESTING")
        # trainer.test(engine, test_loader, ckpt_path="best")

        # TODO: load the best model before return the engine
        return engine, trainer.logger


def prepare_datamodule_cfg(
    datamodule_cfg: DictConfig, dataset: DictConfig, feature_extractor: EdamamePipeline
): ...


def save_results(cfg: DictConfig, results: dict) -> None: ...


def downstream(cfg: DictConfig, pretraine_model: BaseSelfSupervisedEngine) -> None:
    experiment_results: dict[str, dict[int, dict]] = dict()
    for dataset in cfg.datasets:
        datamodule = instantiate(
            prepare_datamodule_cfg(
                cfg.datamodule,
                dataset,
                EdamamePipeline(
                    model=pretraine_model,
                    model_name=None,
                    device=cfg.device,
                    normalization_fn=cfg.normalization_fn,
                    model_config=None,
                ),
            )
        )

        datamodule.extract_features(inplace=True)
        datamodule.train_test_split(inplace=True)
        downstream_engine = instantiate(cfg.engine)
        downstream_engine.fit(datamodule)
        experiment_results[dataset] = downstream_engine.test(datamodule)
    save_results(cfg, experiment_results)


@hydra.main(
    config_path="./configs/train",
    config_name="contrastive_sweep",  # TODO: change config name
    version_base="1.3",
)
def main(cfg: DictConfig):

    pretrained_model, logger = pre_train(cfg=cfg)

    # downstream(cfg=cfg.downstream, pretraine_modele=pretrained_model)

    #############################
    ### FINALIZE LOGGER #########
    #############################
    # Explicitly finalize logger to ensure proper cleanup between runs
    if logger is not None:
        logger.finalize("success")

    # For WandB specifically, ensure run is finished
    if hasattr(logger, "experiment") and hasattr(logger.experiment, "finish"):
        logger.experiment.finish()


if __name__ == "__main__":
    main()
