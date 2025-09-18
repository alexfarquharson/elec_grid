from pathlib import Path

def get_repo_root(marker: str = ".git") -> Path:
    path = Path(__name__).resolve()
    for parent in [path] + list(path.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Repo root not found")

REPO_ROOT = get_repo_root()