import _thread
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from types import TracebackType

import pandas as pd
import psutil

from src import LAB_DIR
from src.experiment._config import StudyConfig, load_study

logger = logging.getLogger(__name__)

_PACKAGES = (
    "anonymous-benchmark", "torch", "numpy", "pandas", "chronos-forecasting", "timesfm",
    "transformers",
)  # fmt: skip

META_FIELDS = (
    "study", "run_id", "status", "exit_reason", "started_utc", "finished_utc", "duration_seconds",
    "host", "os", "cpu_model", "cpu_cores_used", "ram_total_gb", "ram_peak_gb", "gpus",
    "vram_total_gb", "vram_peak_gb", "gpu_hours", "location", "instance_id", "price_per_hour",
    "cost", "workers", "nice", "git_commit", "git_dirty", "python", "versions", "cuda",
    "kronos_commit", "data_version", "seeds", "folds", "plan_hash", "plan_commit",
    "estimated_hours",
)  # fmt: skip
"""Every field of a run's ``meta.json``."""


def _run(command: list[str]) -> str | None:
    executable = shutil.which(command[0])
    if executable is None:
        return None
    done = subprocess.run([executable, *command[1:]], capture_output=True, text=True, check=False)
    return done.stdout.strip() if done.returncode == 0 else None


def _gpus() -> list[dict[str, float | str]]:
    output = _run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"]
    )
    gpus: list[dict[str, float | str]] = []
    for line in (output or "").splitlines():
        name, total, used = (part.strip() for part in line.split(","))
        gpus.append({"name": name, "total_mb": float(total), "used_mb": float(used)})
    return gpus


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _kronos_commit() -> str | None:
    return _run(["git", "-C", str(LAB_DIR / "vendor" / "Kronos"), "rev-parse", "HEAD"])


def _file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def study_results(study_dir: Path) -> Path:
    """Return a study's final results: ``<study>/results/`` (README and the files it cites)."""
    return study_dir / "results"


def study_runs(study_dir: Path) -> Path:
    """Return a study's run outputs: ``<study>/runs/<run-id>/`` (not in git)."""
    return study_dir / "runs"


def study_key(study_dir: Path) -> Path:
    """Return a study's place below ``experiments/`` (``<study>``), or its name.

    ``meta.json`` records it as the study's name. A study folder outside ``experiments/``
    (tests use temporary folders) is keyed by its name alone.
    """
    try:
        return study_dir.resolve().relative_to((LAB_DIR / "experiments").resolve())
    except ValueError:
        return Path(study_dir.name)


def _plan_commit(study_dir: Path) -> str | None:
    readme = str(study_dir / "README.md")
    return _run(["git", "-C", str(LAB_DIR), "log", "-1", "--format=%H", "--", readme])


def _log_to(folder: Path) -> None:
    # every log record of the process also goes to the run's run.log (appended)
    handler = logging.FileHandler(folder / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)


def _log_test_run(path: Path, meta: dict[str, object]) -> None:
    row = pd.DataFrame(
        [{key: meta[key] for key in ("run_id", "started_utc", "plan_hash", "git_commit")}]
    )
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def _cpu_model() -> str:
    for line in (
        Path("/proc/cpuinfo").read_text().splitlines() if Path("/proc/cpuinfo").exists() else []
    ):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


@dataclass
class _Sampler:
    """Samples the peak RAM of this process tree and the peak GPU memory every few seconds."""

    interval: float = 5.0
    budget_seconds: float = float("inf")
    over_budget: bool = False
    ram_peak: float = 0.0
    vram_peak: float = 0.0
    _stop: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._sample()

    def _sample(self) -> None:
        process = psutil.Process()
        rss = process.memory_info().rss + sum(
            c.memory_info().rss for c in process.children(recursive=True)
        )
        self.ram_peak = max(self.ram_peak, rss / 2**30)
        used = sum(float(gpu["used_mb"]) for gpu in _gpus())
        self.vram_peak = max(self.vram_peak, used / 1024)

    def _loop(self) -> None:
        started = pd.Timestamp.now(tz="UTC")
        while not self._stop.wait(self.interval):
            self._sample()
            elapsed = (pd.Timestamp.now(tz="UTC") - started).total_seconds()
            if elapsed > self.budget_seconds and not self.over_budget:
                # the run takes more than twice its estimate: stop it (KeyboardInterrupt in
                # the main thread, so the study closes the run as "stopped")
                self.over_budget = True
                logger.error("run exceeded twice its estimate (%.0f s): stopping it", elapsed)
                _thread.interrupt_main()


def _started_sampler(config: StudyConfig) -> _Sampler:
    # the sampler stops a run that takes more than twice its estimate
    sampler = _Sampler(budget_seconds=2 * config.estimated_hours * 3600)
    sampler.start()
    return sampler


@dataclass(frozen=True, slots=True)
class Placement:
    """Where and how a run is executed.

    Attributes
    ----------
    location : str
        ``"local"`` (the laptop) or ``"vast-ai"`` (a vast.ai instance).
    instance_id : str or None
        The cloud instance, if any.
    price_per_hour : float or None
        Its price in USD per hour, to compute the cost.
    workers : int
        Worker processes or threads the run uses.
    """

    location: str
    instance_id: str | None
    price_per_hour: float | None
    workers: int

    def __post_init__(self) -> None:
        if self.location not in ("local", "vast-ai"):
            raise ValueError(f"location must be 'local' or 'vast-ai', got {self.location!r}")


@dataclass
class Run:
    """One run of a study and the folder it writes to.

    Used as a context manager: an exception inside the ``with`` block closes the run as
    ``"failed"`` (``"stopped"`` for an interrupt, which is also how the runner stops a run over
    twice its estimate) and is raised again; the block itself ends with `finish`.

    Attributes
    ----------
    folder : Path
        ``<study>/runs/<run-id>/`` with ``meta.json``, ``config.toml``, ``run.log``,
        ``metrics.parquet``, ``tables/`` (CSV) and ``figures/`` (PDF from `src.figures`): the
        exact data behind every number of the results README.
    config : StudyConfig
        The run's configuration (its ``config.toml``).
    """

    folder: Path
    config: StudyConfig
    _sampler: _Sampler

    def __enter__(self) -> "Run":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if isinstance(error, KeyboardInterrupt):
            self.finish(
                None, status="stopped", reason="interrupted or over twice the estimated time"
            )
        elif error is not None:
            self.finish(None, status="failed", reason=repr(error))

    def append_metrics(self, metrics: pd.DataFrame) -> None:
        """Close a resumed run's pass: add its rows to ``metrics.parquet``.

        ``meta.json`` keeps the record of the run's first start.
        """
        self._sampler.stop()
        path = self.folder / "metrics.parquet"
        if path.exists():
            metrics = pd.concat([pd.read_parquet(path), metrics], ignore_index=True)
        metrics.to_parquet(path)
        logger.info("run %s: %d metric rows added", self.folder.name, len(metrics))

    def write_tables(self, tables: dict[str, pd.DataFrame]) -> None:
        """Write each table to ``tables/<name>.csv`` of the run, without the index."""
        for name, table in tables.items():
            table.to_csv(self.folder / "tables" / f"{name}.csv", index=False)

    def finish(
        self, metrics: pd.DataFrame | None, *, status: str = "completed", reason: str = ""
    ) -> None:
        """Close the run: write the metrics and the end of ``meta.json``.

        Parameters
        ----------
        metrics : pd.DataFrame or None
            The run's metrics (written to ``metrics.parquet``); None when the run failed.
        status : str, optional
            ``"completed"``, ``"failed"`` or ``"stopped"``.
        reason : str, optional
            Why a run failed or stopped.

        Raises
        ------
        ValueError
            If `status` is not one of the three.
        """
        if status not in ("completed", "failed", "stopped"):
            raise ValueError(f"status must be completed, failed or stopped, got {status!r}")
        self._sampler.stop()
        if metrics is not None:
            metrics.to_parquet(self.folder / "metrics.parquet")
        meta = json.loads((self.folder / "meta.json").read_text())
        finished = pd.Timestamp.now(tz="UTC")
        duration = (finished - pd.Timestamp(meta["started_utc"])).total_seconds()
        n_gpus = len(meta["gpus"]) if meta["device"].startswith("cuda") else 0
        price = meta["price_per_hour"]
        meta.update({
            "status": status, "exit_reason": reason, "finished_utc": finished.isoformat(),
            "duration_seconds": round(duration, 1), "ram_peak_gb": round(self._sampler.ram_peak, 2),
            "vram_peak_gb": round(self._sampler.vram_peak, 2),
            "gpu_hours": round(n_gpus * duration / 3600, 3),
            "cost": None if price is None else round(price * duration / 3600, 4),
        })  # fmt: skip
        (self.folder / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("run %s %s after %.0f s", meta["run_id"], status, duration)


def start_run(
    study_dir: Path,
    run_id: str,
    config: StudyConfig,
    *,
    data_dir: Path,
    placement: Placement,
) -> Run:
    """Create the folder of a new run and record what produced it in ``meta.json``.

    Parameters
    ----------
    study_dir : Path
        ``experiments/<study>``.
    run_id : str
        A new, unique id (e.g. ``2026-11-16T0800``).
    config : StudyConfig
        The study's configuration (seeds and device are recorded).
    data_dir : Path
        The lab's ``data/`` folder (``data/manifest.json`` gives the data version). The run goes
        to ``<study>/runs/<run_id>/`` (`study_runs`).
    placement : Placement
        Where it runs, its price and its workers.

    Returns
    -------
    Run
        The run; logging also goes to its ``run.log``, and a sampler records peak RAM and VRAM.

    Raises
    ------
    FileExistsError
        If the run folder exists: a finished run is never overwritten.
    """
    if config.folds == "test" and not config.plan_locked:
        raise PermissionError(
            f"{study_dir.name}: test folds need a locked plan ([plan] locked = true)"
        )
    folder = study_runs(study_dir) / run_id
    if folder.exists():
        raise FileExistsError(f"run {folder} exists; choose a new run id")
    (folder / "figures").mkdir(parents=True)
    (folder / "tables").mkdir()
    shutil.copy(study_dir / "study.toml", folder / "config.toml")
    manifest = data_dir / "manifest.json"
    gpus = _gpus()
    status = _run(["git", "-C", str(LAB_DIR), "status", "--porcelain"])
    meta: dict[str, object] = {
        "study": str(study_key(study_dir)), "run_id": run_id, "status": "running",
        "exit_reason": "",
        "started_utc": pd.Timestamp.now(tz="UTC").isoformat(), "finished_utc": None,
        "duration_seconds": None, "host": platform.node(), "os": platform.platform(),
        "cpu_model": _cpu_model(), "cpu_cores_used": len(os.sched_getaffinity(0)),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1), "ram_peak_gb": None,
        "gpus": [gpu["name"] for gpu in gpus],
        "vram_total_gb": round(sum(float(gpu["total_mb"]) for gpu in gpus) / 1024, 1),
        "vram_peak_gb": None, "gpu_hours": None, "device": config.device,
        "location": placement.location, "instance_id": placement.instance_id,
        "price_per_hour": placement.price_per_hour, "cost": None,
        "workers": placement.workers, "nice": os.nice(0),
        "git_commit": _run(["git", "-C", str(LAB_DIR), "rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
        "python": platform.python_version(), "versions": _versions(),
        "cuda": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "kronos_commit": _kronos_commit(),
        "data_version": _file_hash(manifest),
        "seeds": list(config.seeds), "folds": config.folds,
        "plan_hash": _file_hash(study_dir / "README.md"),
        "plan_commit": _plan_commit(study_dir),
        "estimated_hours": config.estimated_hours,
    }  # fmt: skip
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    if config.folds == "test":
        _log_test_run(study_runs(study_dir) / "test-log.csv", meta)
    _log_to(folder)
    logger.info("run %s of %s started in %s", run_id, study_dir.name, folder)
    return Run(folder, config, _started_sampler(config))


def resume_run(study_dir: Path, run_id: str, *, data_dir: Path) -> Run:
    """Reopen an existing run to add outputs to it, with the configuration it started with.

    A run is resumed only from the code, the plan and the data it started with, so every
    output of the run has one record in its ``meta.json`` (which keeps the first start). Each
    resumed pass has its own time budget (twice the estimate) and adds its rows to
    ``metrics.parquet`` (`Run.append_metrics`).

    Parameters
    ----------
    study_dir : Path
        ``experiments/<study>``.
    run_id : str
        The existing run, ``<study>/runs/<run_id>/``.
    data_dir : Path
        The lab's ``data/`` folder (``data/manifest.json`` gives the data version).

    Returns
    -------
    Run
        The run with its own ``config.toml`` (not the study's current ``study.toml``);
        logging also appends to its ``run.log``.

    Raises
    ------
    ValueError
        If the git commit, the dirty flag of the work tree, the plan's hash or the data
        version differs from the run's start.
    """
    folder = study_runs(study_dir) / run_id
    meta = json.loads((folder / "meta.json").read_text())
    status = _run(["git", "-C", str(LAB_DIR), "status", "--porcelain"])
    now = {
        "git_commit": _run(["git", "-C", str(LAB_DIR), "rev-parse", "HEAD"]),
        "git_dirty": None if status is None else bool(status),
        "plan_hash": _file_hash(study_dir / "README.md"),
        "data_version": _file_hash(data_dir / "manifest.json"),
    }
    for key, value in now.items():
        if meta[key] != value:
            raise ValueError(
                f"run {run_id} started with {key} {meta[key]}, now {value}: start a new run"
            )
    _log_to(folder)
    logger.info("resuming run %s of %s", run_id, study_dir.name)
    config = load_study(folder / "config.toml")
    return Run(folder, config, _started_sampler(config))
