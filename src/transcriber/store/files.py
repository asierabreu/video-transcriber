"""Per-job file layout (§8.2) and atomic writes."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from transcriber.ids import is_valid_job_id


class JobFiles:
    """Paths inside one job directory."""

    def __init__(self, jobs_dir: Path, job_id: str) -> None:
        if not is_valid_job_id(job_id):
            raise ValueError(f"invalid job id: {job_id!r}")
        self.job_id = job_id
        self.dir = jobs_dir / job_id
        self.out_dir = self.dir / "out"

    def path(self, name: str) -> Path:
        """Path of an artifact in the job directory."""
        return self.dir / name

    @property
    def upload_part(self) -> Path:
        return self.dir / "upload.part"

    @property
    def audio(self) -> Path:
        return self.dir / "audio.wav"

    def source(self) -> Path | None:
        """The stored source video, if present."""
        matches = sorted(self.dir.glob("source.*"))
        return matches[0] if matches else None

    def output(self, fmt: str) -> Path:
        """Rendered transcript in the given format."""
        return self.out_dir / f"transcript.{fmt}"

    def exists(self, name: str) -> bool:
        return (self.dir / name).exists()

    def read_json(self, name: str) -> Any:
        return json.loads((self.dir / name).read_text(encoding="utf-8"))

    def write_json(self, name: str, data: Any) -> Path:
        return atomic_write_text(self.dir / name, json.dumps(data, ensure_ascii=False, indent=1))

    def write_text(self, name: str, text: str) -> Path:
        return atomic_write_text(self.dir / name, text)

    def delete_media(self) -> None:
        """Remove the source video and extracted audio."""
        for p in [self.source(), self.audio, self.upload_part]:
            if p is not None and p.exists():
                p.unlink()

    def delete_all(self) -> None:
        """Remove the whole job directory."""
        shutil.rmtree(self.dir, ignore_errors=True)


def atomic_write_text(path: Path, text: str) -> Path:
    """Write `text` to `path` via a temp file and os.replace, so readers never see partial data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def free_disk_gb(path: Path) -> float:
    """Free space on the filesystem holding `path`, in GiB."""
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 2**30
