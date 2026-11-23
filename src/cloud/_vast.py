import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SSH_OPTIONS = ["-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30"]


@dataclass(frozen=True, slots=True)
class Offer:
    """A rentable vast.ai machine.

    Attributes
    ----------
    id : int
        The offer id.
    gpu : str
        GPU model.
    vram_gb : float
        GPU memory.
    price_per_hour : float
        Total price in USD per hour.
    location : str
        Where the machine is.
    """

    id: int
    gpu: str
    vram_gb: float
    price_per_hour: float
    location: str


def _vastai(*arguments: str, answer: str = "") -> object:
    """Run the vastai command line tool (its API key lives in ~/.config/vastai) and parse JSON.

    `answer` is typed into a confirmation prompt (``destroy`` asks "are you sure").
    """
    executable = shutil.which("vastai") or str(Path(sys.executable).parent / "vastai")
    done = subprocess.run(  # noqa: S603 (fixed tool, arguments built here)
        [executable, *arguments, "--raw"], input=answer, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        raise RuntimeError(f"vastai {arguments[0]} failed: {done.stderr.strip()[-300:]}")
    text = done.stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # some commands (e.g. create ssh-key) answer in plain text


def cheapest_offer(*, max_price: float, min_vram_gb: float, min_cuda: float) -> Offer:
    """Return the cheapest reliable single-GPU offer under the price cap.

    Raises
    ------
    LookupError
        If no offer meets the limits.
    """
    query = (
        f"num_gpus=1 gpu_ram>={min_vram_gb} dph_total<={max_price} cuda_max_good>={min_cuda} "
        "reliability>0.98 rentable=true inet_down>=200 disk_space>=40"
    )
    offers = _vastai("search", "offers", query, "-o", "dph")
    if not offers:
        raise LookupError(f"no offer for {query}")
    best = offers[0]  # type: ignore[index]
    return Offer(
        int(best["id"]), best["gpu_name"], round(best["gpu_ram"] / 1024, 1),
        float(best["dph_total"]), str(best.get("geolocation")),
    )  # fmt: skip


def ensure_ssh_key(public_key: Path) -> None:
    """Register the local public SSH key with the vast.ai account unless it is already there."""
    key = public_key.read_text().strip()
    known = _vastai("show", "ssh-keys") or []
    if not any(entry.get("public_key", "").strip() == key for entry in known):  # type: ignore[union-attr]
        _vastai("create", "ssh-key", key)
        logger.info("registered %s with vast.ai", public_key)


def create_instance(offer: Offer, *, image: str, disk_gb: int, guard_minutes: int) -> int:
    """Rent `offer` and return the instance id.

    The instance shuts itself down after `guard_minutes`, a second guard next to the
    caller's own destroy.
    """
    onstart = f"(sleep {guard_minutes * 60}; poweroff || kill 1) >/dev/null 2>&1 &"
    created = _vastai(
        "create", "instance", str(offer.id), "--image", image, "--disk", str(disk_gb),
        "--ssh", "--direct", "--onstart-cmd", onstart,
    )  # fmt: skip
    instance = int(created["new_contract"])  # type: ignore[index]
    logger.info("rented instance %d (%s, $%.3f/h)", instance, offer.gpu, offer.price_per_hour)
    return instance


def wait_for_ssh(instance: int, *, timeout_seconds: float) -> tuple[str, int]:
    """Wait until the instance runs and accepts SSH, and return its SSH host and port.

    Raises
    ------
    TimeoutError
        If it is not running in time.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        details = _vastai("show", "instance", str(instance))
        if details and details.get("actual_status") == "running" and details.get("ssh_host"):  # type: ignore[union-attr]
            host, port = str(details["ssh_host"]), int(details["ssh_port"])  # type: ignore[index]
            if ssh(host, port, "true", timeout_seconds=30).returncode == 0:
                return host, port
        time.sleep(15)
    raise TimeoutError(f"instance {instance} not running after {timeout_seconds:.0f} s")


def destroy_instance(instance: int, *, timeout_seconds: float = 120) -> None:
    """Destroy the instance (stops all charges) and wait until the account no longer lists it.

    Raises
    ------
    RuntimeError
        If the instance is still listed after `timeout_seconds`.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        _vastai("destroy", "instance", str(instance), answer="y\n")
        time.sleep(10)
        if all(int(entry["id"]) != instance for entry in running_instances()):  # type: ignore[arg-type]
            logger.info("destroyed instance %d", instance)
            return
    raise RuntimeError(f"instance {instance} is still listed: destroy it in the vast.ai console")


def running_instances() -> list[dict[str, object]]:
    """Return every instance of the account (should be empty after a run)."""
    return list(_vastai("show", "instances") or [])  # type: ignore[arg-type]


def ssh(
    host: str, port: int, command: str, *, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Run `command` on the instance as root and return the finished process (output captured)."""
    return subprocess.run(  # noqa: S603
        ["ssh", *_SSH_OPTIONS, "-p", str(port), f"root@{host}", command],  # noqa: S607
        capture_output=True, text=True, check=False, timeout=timeout_seconds,
    )  # fmt: skip


def rsync(source: str, target: str, port: int, *, excludes: list[str]) -> None:
    """Copy files to or from the instance with rsync over SSH.

    Raises
    ------
    RuntimeError
        If rsync fails.
    """
    command = ["rsync", "-az", "-e", f"ssh {' '.join(_SSH_OPTIONS)} -p {port}"]
    command += [f"--exclude={pattern}" for pattern in excludes] + [source, target]
    done = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if done.returncode != 0:
        raise RuntimeError(f"rsync failed: {done.stderr.strip()[-300:]}")
