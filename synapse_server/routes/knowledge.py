"""Knowledge base management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

router = APIRouter()


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
    content = await file.read()
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
