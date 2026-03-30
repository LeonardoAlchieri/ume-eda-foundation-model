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
) -> Path | None:
    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed. Cannot download checkpoint from WandB.")
        return None

    api = wandb.Api()
    candidate_entities = [
        entity,
        os.getenv("WANDB_ENTITY"),
        getattr(api, "default_entity", None),
    ]
    try:
        viewer = api.viewer
        if isinstance(viewer, dict):
            candidate_entities.append(viewer.get("entity"))
    except Exception:
        pass

    candidate_run_paths = [
        f"{candidate_entity}/{project}/{run_id}"
        for candidate_entity in candidate_entities
        if candidate_entity
    ]
    candidate_run_paths.append(f"{project}/{run_id}")

    run = None
    for run_path in dict.fromkeys(candidate_run_paths):
        try:
            run = api.run(run_path)
            logger.info(f"Found WandB run for resume at: {run_path}")
            break
        except Exception:
            continue

    if run is None:
        logger.warning(
            "Could not access the requested WandB run. Skipping WandB checkpoint download."
        )
        return None

    resume_dir = Path(save_dir).expanduser() / "resume_checkpoints" / run_id
    resume_dir.mkdir(parents=True, exist_ok=True)

    # First try the canonical run file name.
    try:
        run_file = run.file("last.ckpt")
        if run_file is not None:
            run_file.download(root=str(resume_dir), replace=True)
            candidate = resume_dir / "last.ckpt"
            if candidate.is_file():
                logger.info(f"Downloaded checkpoint from WandB run files: {candidate}")
                return candidate
    except Exception:
        pass

    # Then try checkpoint artifacts logged by WandB logger.
    artifacts = []
    try:
        artifacts.extend(list(run.logged_artifacts()))
    except Exception:
        pass

    for artifact in artifacts:
        try:
            artifact_dir = Path(artifact.download(root=str(resume_dir)))
            checkpoint_candidates = list(artifact_dir.rglob("*.ckpt"))
            best_candidate = _pick_best_checkpoint(checkpoint_candidates, run_id=run_id)
            if best_candidate is not None and best_candidate.is_file():
                logger.info(
                    f"Downloaded checkpoint from WandB artifact {artifact.name}: {best_candidate}"
                )
                return best_candidate
        except Exception:
            continue

    return None


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
    logger_cfg = cfg.get("logger")
    if logger_cfg is None:
        return None

    logger_target = str(logger_cfg.get("_target_", ""))
    if "WandbLogger" not in logger_target:
        return None

    resume_mode = str(logger_cfg.get("resume", "")).lower()
    if resume_mode != "must":
        return None

    run_id = str(logger_cfg.get("id", "") or "").strip()
    if not run_id:
        raise ValueError("WandB resume='must' requires logger.id to be set.")

    project = str(logger_cfg.get("project", "") or "").strip()
    if not project:
        raise ValueError("WandB resume='must' requires logger.project to be set.")

    save_dir = str(logger_cfg.get("save_dir", "./wandb_logs.nosync"))
    entity = logger_cfg.get("entity")

    checkpoint_path = _download_checkpoint_from_wandb(
        project=project,
        run_id=run_id,
        save_dir=save_dir,
        entity=entity,
    )
    if checkpoint_path is not None:
        return str(checkpoint_path)

    checkpoint_path = _find_checkpoint_in_local_lightning_cache(
        run_id=run_id,
        save_dir=save_dir,
    )
    if checkpoint_path is not None:
        return str(checkpoint_path)

    raise FileNotFoundError(
        "WandB resume='must' is enabled but no checkpoint could be recovered from "
        "WandB or local Lightning cache."
    )
