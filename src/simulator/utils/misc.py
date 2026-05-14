"""
Miscellaneous utils for the simulator package.
"""

from __future__ import annotations

import random
import string
import subprocess
from typing import Any

import yaml


def random_string(length: int = 8) -> str:
    """Return a random alphanumeric string of the given length."""
    return "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))


def yaml_safe_load(fname: str) -> dict[str, Any]:
    """Open a YAML file and return it as a dict."""
    with open(fname, "r", encoding="UTF-8") as f:
        return yaml.safe_load(f)


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_branch() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("ascii")
        .strip()
    )


def ensure_repo_is_clean(repo_path: str = ".", new_files_ok: bool = True) -> None:
    """Raise if the git repo at repo_path has uncommitted changes."""
    try:
        result = subprocess.check_output(
            ["git", "-C", repo_path, "status", "--porcelain"], text=True
        ).splitlines()

        if new_files_ok:
            result = [line for line in result if not line.startswith("??")]

        if result:
            raise ValueError("The repository is not clean. Please commit or stash your changes.")
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            "Error checking the repository status. Make sure the path provided is a valid Git "
            "repository."
        ) from exc


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten a nested dict by joining nested keys with `sep`."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
