import sys
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from .config import CODE_DIR, CHECKPOINT_PATH

jobs: Dict[str, "JobState"] = {}


@dataclass
class JobState:
    job_id: str
    status: Literal["queued", "processing", "done", "error"]
    targets: List[str]
    input_path: str
    output_dir: str
    completed: int = 0
    current_target: Optional[str] = None
    error: Optional[str] = None
    results: List[dict] = field(default_factory=list)


def process_job(job_id: str) -> None:
    job = jobs[job_id]
    job.status = "processing"

    for target in job.targets:
        job.current_target = target
        output_path = f"{job.output_dir}/{target}.wav"

        cmd = [
            sys.executable,
            str(CODE_DIR / "inference.py"),
            "--audio", job.input_path,
            "--output", output_path,
            "--target", target,
            "--checkpoint", str(CHECKPOINT_PATH),
            "--device", "cpu",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(CODE_DIR),
        )

        if result.returncode != 0:
            job.status = "error"
            job.error = result.stderr or f"Separation failed for target '{target}'"
            return

        job.completed += 1
        job.results.append({
            "target": target,
            "download_url": f"/api/download/{job_id}/{target}",
        })

    job.status = "done"
    job.current_target = None
