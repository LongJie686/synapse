"""File upload and image reading endpoints."""

from __future__ import annotations

import re
import uuid
import base64
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()

_project_root = Path(__file__).resolve().parent.parent.parent
_upload_dir = _project_root / "data" / "uploads"

# UUID + optional extension — matches stored filenames like "<uuid4>.<ext>"
_SAFE_FILE_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(\.[a-zA-Z0-9]{1,10})?$"
)


_ALLOWED_EXTENSIONS = {
    # images
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp",
    # documents
    ".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx", ".json",
}


def _ensure_upload_dir() -> Path:
    _upload_dir.mkdir(parents=True, exist_ok=True)
    return _upload_dir


def _resolve_safe_path(file_id: str) -> Path:
    """Resolve file path and guard against path traversal."""
    if not _SAFE_FILE_ID.match(file_id):
        raise HTTPException(status_code=400, detail="Invalid file ID format")
    resolved = (_upload_dir / file_id).resolve()
    try:
        resolved.relative_to(_upload_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    return resolved


@router.post("/files/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """Upload a file and return its metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")

    _ensure_upload_dir()
    ext = Path(file.filename).suffix.lower()
    file_id = str(uuid.uuid4())
    stored_name = f"{file_id}{ext}"
    file_path = _upload_dir / stored_name

    with open(file_path, "wb") as f:
        f.write(content)

    is_image = file.content_type and file.content_type.startswith("image/") if file.content_type else False
    mime_type = file.content_type or "application/octet-stream"

    return {
        "id": file_id,
        "filename": file.filename,
        "stored_name": stored_name,
        "size": len(content),
        "mime_type": mime_type,
        "is_image": is_image,
        "url": f"/api/files/{stored_name}/download",
    }


@router.get("/files/{file_id}")
async def get_file(file_id: str) -> dict:
    """Get file metadata."""
    file_path = _resolve_safe_path(file_id)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "id": file_id,
        "stored_name": file_id,
        "size": file_path.stat().st_size,
        "url": f"/uploads/{file_id}",
    }


@router.get("/files/{file_id}/download")
async def download_file(file_id: str) -> FileResponse:
    """Download a file by its stored name."""
    file_path = _resolve_safe_path(file_id)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=file_id)


@router.delete("/files/{file_id}")
async def delete_file(file_id: str) -> dict:
    """Delete an uploaded file."""
    file_path = _resolve_safe_path(file_id)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    file_path.unlink()
    return {"id": file_id, "status": "deleted"}


@router.post("/files/read-image")
async def read_image(file: UploadFile = File(...)) -> dict:
    """Read an image file and return base64 encoded data with description."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")

    mime_type = file.content_type or "image/png"
    b64 = base64.b64encode(content).decode("utf-8")

    return {
        "filename": file.filename,
        "mime_type": mime_type,
        "size": len(content),
        "base64": b64,
        "data_url": f"data:{mime_type};base64,{b64}",
    }
