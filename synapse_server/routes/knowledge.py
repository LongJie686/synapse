"""Knowledge base management endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()

_ALLOWED_DOC_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv"}
_MAX_DOC_SIZE = 50 * 1024 * 1024  # 50MB


class CollectionCreateRequest(BaseModel):
    name: str
    description: str = ""


class QueryRequest(BaseModel):
    query: str
    collection: str = ""
    strategy: str = "basic"
    top_k: int = 5


@router.get("/knowledge/collections")
async def list_collections() -> list[dict]:
    return []


@router.post("/knowledge/collections")
async def create_collection(request: CollectionCreateRequest) -> dict:
    return {"name": request.name, "status": "created"}


@router.post("/knowledge/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not allowed. Allowed: {', '.join(sorted(_ALLOWED_DOC_EXTENSIONS))}",
        )

    content = await file.read()
    if len(content) > _MAX_DOC_SIZE:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")

    return {
        "filename": file.filename,
        "size": len(content),
        "status": "uploaded",
    }


@router.post("/knowledge/query")
async def query_knowledge(request: QueryRequest) -> dict:
    return {
        "query": request.query,
        "results": [],
        "strategy": request.strategy,
    }
