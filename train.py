from logging import getLogger
import json
from pathlib import Path
from functools import partial

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader

from edamame.engine import BaseSelfSupervisedEngine
from edamame_downstream.feature_extraction.edamame import EdamameExtractor
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
    except Exception as e:
        logger.error(f"Training failed with exception: {e}")
        raise e
    # finally:
    #     print("NO TESTING")
    #     # trainer.test(engine, test_loader, ckpt_path="best")

    #     # TODO: load the best model before return the engine
    return engine, trainer.logger


def prepare_datamodule_cfg(
    datamodule_cfg: DictConfig, dataset: DictConfig, segment_length_in_s: int
):
    datamodule_cfg = datamodule_cfg.copy()
    datamodule_cfg.path_to_data = f"{datamodule_cfg.path_to_data}/{dataset.name}/data_{dataset.side}_{dataset.label_name}_{segment_length_in_s}s.npz"
    datamodule_cfg.feature_extractor = None
    datamodule_cfg.name = dataset.name
    return datamodule_cfg


def save_results(
    cfg: DictConfig,
    results: dict[
        tuple[
            str,
            str,
            str,
        ],
        dict,
    ],
) -> None:
    """Save downstream task results to JSON file."""
    output_dir = Path(cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    results_to_be_saved: dict[str, dict] = {
        f"{key[0]}_{key[1]}_{key[2]}": dataset_results
        for key, dataset_results in results.items()
    }

    output_file = output_dir / "downstream_results.json"
    with open(output_file, "w") as f:
        json.dump(results_to_be_saved, f, indent=2, default=str)

    logger.info(f"Results saved to {output_file}")


def downstream(
    cfg: DictConfig,
    pretraine_model: BaseSelfSupervisedEngine,
    segment_length_in_s: int,
    lightning_logger: Logger,
) -> None:
    experiment_results: dict[str, dict[int, dict]] = dict()
    for dataset in cfg.datasets:
        dataset: DictConfig
        datamodule = instantiate(
            prepare_datamodule_cfg(
                datamodule_cfg=cfg.datamodule,
                dataset=dataset,
                segment_length_in_s=segment_length_in_s,
            )
        )
        datamodule.set_feature_extractor(
            EdamameExtractor(
                pipeline=pretraine_model,
                model_name="mae",  # TODO: make this configurable
                checkpoint_path=None,
                device_map=cfg.device,
                # normalization_fn=cfg.normalization_fn,
                batch_size=1024,
            )
        )

        datamodule.extract_features(inplace=True)
        datamodule.train_test_split(inplace=True)
        downstream_engine = instantiate(cfg.engine)
        downstream_engine.fit(datamodule)
        experiment_results[tuple(dataset.values())] = downstream_engine.test(
            datamodule, lightning_logger=lightning_logger
        )
    save_results(cfg, experiment_results)


@hydra.main(
    config_path="./configs",
    config_name="conf2",  # TODO: change config name
    version_base="1.3",
)
def main(cfg: DictConfig):

    pretrained_model, lightning_logger = pre_train(cfg=cfg.pretrain)

    try:
        downstream(
            cfg=cfg.downstream,
            pretraine_model=pretrained_model,
            segment_length_in_s=cfg.segment_length // cfg.sampling_rate,
            lightning_logger=lightning_logger,
        )
    except Exception as e:
        logger.error(f"Downstream task failed with exception: {e}")
        raise e
    finally:
        #############################
        ### FINALIZE LOGGER #########
        #############################
        # Explicitly finalize logger to ensure proper cleanup between runs
        if lightning_logger is not None:
            lightning_logger.finalize("success")

        # For WandB specifically, ensure run is finished
        if hasattr(lightning_logger, "experiment") and hasattr(
            lightning_logger.experiment, "finish"
        ):
            lightning_logger.experiment.finish()


if __name__ == "__main__":
    main()
