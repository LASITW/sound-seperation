import json
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import (
    ALLOWED_EXTENSIONS,
    BASE_DIR,
    MAX_FILE_SIZE_MB,
    OUTPUTS_DIR,
    UPLOADS_DIR,
    VALID_TARGETS,
)
from .worker import JobState, jobs, process_job

app = FastAPI(title="Sound Separation API")

# Serve frontend at root
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend"), html=True), name="static")


@app.on_event("startup")
def startup():
    UPLOADS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
def devtools():
    return Response(status_code=204)


@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "frontend" / "index.html"))


@app.post("/api/separate", status_code=202)
async def separate(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    targets: str = Form(...),
):
    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use WAV, MP3, or FLAC.")

    # Parse and validate targets
    try:
        target_list: list[str] = json.loads(targets)
    except Exception:
        raise HTTPException(400, "targets must be a JSON array string.")

    if not target_list or not set(target_list).issubset(VALID_TARGETS):
        raise HTTPException(400, f"targets must be a non-empty subset of {sorted(VALID_TARGETS)}.")

    # Read file and check size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")

    # Save upload
    job_id = str(uuid.uuid4())
    input_path = UPLOADS_DIR / f"{job_id}{suffix}"
    input_path.write_bytes(content)

    # Create output directory
    output_dir = OUTPUTS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Register job
    jobs[job_id] = JobState(
        job_id=job_id,
        status="queued",
        targets=target_list,
        input_path=str(input_path),
        output_dir=str(output_dir),
    )

    background_tasks.add_task(process_job, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "targets": target_list,
        "message": f"Job queued. Poll /api/status/{job_id} for updates.",
    }


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    response: dict = {"job_id": job_id, "status": job.status}

    if job.status == "processing":
        response["progress"] = {
            "completed": job.completed,
            "total": len(job.targets),
            "current_target": job.current_target,
        }
    elif job.status == "done":
        response["results"] = job.results
    elif job.status == "error":
        response["error"] = job.error

    return response


@app.get("/api/download/{job_id}/{target}")
def download(job_id: str, target: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    if target not in VALID_TARGETS:
        raise HTTPException(400, f"Invalid target '{target}'.")

    output_path = Path(job.output_dir) / f"{target}.wav"
    if not output_path.exists():
        raise HTTPException(404, f"Output for '{target}' not ready or not requested.")

    return FileResponse(
        str(output_path),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{target}.wav"'},
    )
