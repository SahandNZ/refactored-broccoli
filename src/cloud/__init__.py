"""Renting GPUs on vast.ai through its command line tool (the key stays in ~/.config/vastai)."""

from src.cloud._vast import (
    Offer,
    cheapest_offer,
    create_instance,
    destroy_instance,
    ensure_ssh_key,
    rsync,
    running_instances,
    ssh,
    wait_for_ssh,
)

__all__ = [
    "Offer",
    "cheapest_offer",
    "create_instance",
    "destroy_instance",
    "ensure_ssh_key",
    "rsync",
    "running_instances",
    "ssh",
    "wait_for_ssh",
]
