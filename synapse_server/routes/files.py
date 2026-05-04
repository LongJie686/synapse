"""File upload and image reading endpoints."""

from __future__ import annotations

import os
import uuid
import base64
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

router = APIRouter()

_project_root = Path(__file__).resolve().parent.parent.parent
_upload_dir = _project_root / "data" / "uploads"


def _ensure_upload_dir() -> Path:
    _upload_dir.mkdir(parents=True, exist_ok=True)
    return _upload_dir


@router.post("/files/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    """Upload a file and return its metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Limit file size to 20MB
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 20MB)")

    _ensure_upload_dir()
    ext = Path(file.filename).suffix or ""
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
        "url": f"/uploads/{stored_name}",
    }


@router.get("/files/{file_id}")
async def get_file(file_id: str) -> dict:
    """Get file metadata."""
    file_path = _upload_dir / file_id
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return {
        "id": file_id,
        "stored_name": file_id,
        "size": file_path.stat().st_size,
        "url": f"/uploads/{file_id}",
    }


@router.delete("/files/{file_id}")
async def delete_file(file_id: str) -> dict:
    """Delete an uploaded file."""
    file_path = _upload_dir / file_id
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
