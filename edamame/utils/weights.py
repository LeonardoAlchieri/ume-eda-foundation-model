from logging import getLogger
import os
from pathlib import Path

from omegaconf import DictConfig

logger = getLogger(__name__)


def _pick_best_checkpoint(
    checkpoint_paths: list[Path], run_id: str | None = None
) -> Path | None:
    if not checkpoint_paths:
        return None

    def _score(path: Path) -> tuple[int, int, float]:
        path_str = str(path)
        return (
            0 if path.name == "last.ckpt" else 1,
            0 if run_id and run_id in path_str else 1,
            -path.stat().st_mtime,
        )

    return min(checkpoint_paths, key=_score)


def _download_checkpoint_from_wandb(
    project: str,
    run_id: str,
    save_dir: str,
    entity: str | None = None,
) -> Path:
    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed. Cannot download checkpoint from WandB.")
        return None
    
    api = wandb.Api()
    UserWarning("This API currently works for artifacts saved manually. Will be updated in the future to support automatically saved checkpoints.")
    artifact = api.artifact(f"{entity}/{project}/model-{run_id}:v0")
    model_state = artifact.get_entry("last.ckpt")
    resume_dir = Path(save_dir).expanduser() / "resume_checkpoints" / run_id
    resume_dir.mkdir(parents=True, exist_ok=True)
    return model_state.download(root=str(resume_dir))


def _find_checkpoint_in_local_lightning_cache(
    run_id: str,
    save_dir: str,
) -> Path | None:
    candidate_dirs = [
        Path(save_dir).expanduser(),
        Path.cwd() / "lightning_logs",
        Path.cwd() / "lightning_logs.nosync",
        Path.cwd() / "checkpoints",
        Path.cwd() / "checkpoints.nosync",
        Path.home() / ".lightning",
        Path.home() / ".cache" / "torch" / "lightning",
    ]

    for base_dir in candidate_dirs:
        if not base_dir.exists():
            continue

        checkpoint_candidates = list(base_dir.rglob(f"*{run_id}*.ckpt"))
        if not checkpoint_candidates:
            checkpoint_candidates = list(base_dir.rglob("last.ckpt"))

        best_candidate = _pick_best_checkpoint(checkpoint_candidates, run_id=run_id)
        if best_candidate is not None and best_candidate.is_file():
            logger.info(
                f"Using checkpoint from local Lightning cache: {best_candidate}"
            )
            return best_candidate

    return None


def _resolve_resume_checkpoint_path(cfg: DictConfig) -> str | None:
    logger_cfg = cfg["logger"]

    logger_target = str(logger_cfg["_target_"])
    if "WandbLogger" not in logger_target:
        raise NotImplementedError(
            f"Unsupported logger type for resume='must' mode: {logger_target}. Only WandbLogger is currently supported."
        )

    resume_mode = str(logger_cfg["resume"]).lower()
    if resume_mode != "must":
        return None

    run_id = str(logger_cfg["id"] or "").strip()
    if not run_id:
        raise ValueError("WandB resume='must' requires logger.id to be set.")

    project = str(logger_cfg["project"] or "").strip()
    if not project:
        raise ValueError("WandB resume='must' requires logger.project to be set.")

    save_dir = str(logger_cfg["save_dir"])
    
    entity = logger_cfg["entity"]

    checkpoint_path = _download_checkpoint_from_wandb(
        project=project,
        run_id=run_id,
        save_dir=save_dir,
        entity=entity,
    )
    if checkpoint_path is not None:
        return str(checkpoint_path)

    raise FileNotFoundError(
        "WandB resume='must' is enabled but checkpoint could not be downloaded from WandB. "
        "Local Lightning cache lookup is currently disabled for resume='must' mode to avoid "
        "accidentally picking up the wrong checkpoint. Please ensure that the correct checkpoint "
        "is available in WandB and try again."
    )
    # checkpoint_path = _find_checkpoint_in_local_lightning_cache(
    #     run_id=run_id,
    #     save_dir=save_dir,
    # )
    # if checkpoint_path is not None:
    #     return str(checkpoint_path)

    # raise FileNotFoundError(
    #     "WandB resume='must' is enabled but no checkpoint could be recovered from "
    #     "WandB or local Lightning cache."
    # )
