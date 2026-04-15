from logging import getLogger
import json
from pathlib import Path
from functools import partial

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import Logger
import torch
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
    print("NO TESTING")
    # trainer.test(engine, test_loader, ckpt_path="best")


    if getattr(trainer, "checkpoint_callback", None) is not None:
        best_model_path = trainer.checkpoint_callback.best_model_path

    if not best_model_path:
        for callback in getattr(trainer, "callbacks", []):
            if getattr(callback, "best_model_path", ""):
                best_model_path = callback.best_model_path
                break

    if best_model_path and Path(best_model_path).is_file():
        checkpoint = torch.load(best_model_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing_keys, unexpected_keys = engine.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            logger.warning(
                "Loaded best checkpoint with non-strict matching. "
                f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
            )
        logger.info(f"Loaded best model weights from checkpoint: {best_model_path}")
    else:
        logger.warning(
            "No best checkpoint found. Returning engine with current in-memory weights."
        )
    
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
    model_name: str,
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
                model_name=model_name,  # TODO: make this configurable
                checkpoint_path=None,
                device_map=cfg.device,
                # normalization_fn=cfg.normalization_fn,
                batch_size=cfg.batch_size,  
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
    version_base="1.3",
)
def main(cfg: DictConfig):

    pretrained_model, lightning_logger = pre_train(cfg=cfg.pretrain)

    try:
        NotImplementedError("Downstream task not implemented yet for multi-modal pretraining.")
        # downstream(
        #     cfg=cfg.downstream,
        #     pretraine_model=pretrained_model,
        #     segment_length_in_s=cfg.segment_length // cfg.sampling_rate,
        #     lightning_logger=lightning_logger,
        #     model_name=cfg.pretrain.engine.model._target_.split(".")[-1],
        # )
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
