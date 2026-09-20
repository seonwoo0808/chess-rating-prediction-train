"""File identity and preprocessing version for restart validation."""
import hashlib
from pathlib import Path


def dataset_manifest(paths, **config):
    files = []
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        files.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    preprocessing = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {"files": files, "config": config, "preprocessing": preprocessing}
