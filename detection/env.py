import os
from pathlib import Path


def configure_matplotlib_env(project_root: Path) -> None:
    cache_dir = project_root / ".cache"
    cache_dir.mkdir(exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    matplotlib_cache_dir = cache_dir / "matplotlib"
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
