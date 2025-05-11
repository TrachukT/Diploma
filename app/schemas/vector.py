from typing import List, Dict, Any, Optional

from pydantic import BaseModel

from app.services.vector_db import Document


# Моделі запитів
class DocumentWriteRequest(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    id: Optional[str] = None


class BatchWriteRequest(BaseModel):
    documents: List[DocumentWriteRequest]


class DocumentUpdateRequest(BaseModel):
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    include_embeddings: bool = False


class HybridSearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    alpha: float = 0.5


class DeleteRequest(BaseModel):
    document_ids: Optional[List[str]] = None
    delete_all: bool = False


class DeleteByFiltersRequest(BaseModel):
    filters: Dict[str, Any]


# Моделі відповідей
class DocumentResponse(BaseModel):
    documents: List[Document]


class DocumentIdResponse(BaseModel):
    document_id: str


class DocumentsWrittenResponse(BaseModel):
    documents_written: int


class StatusResponse(BaseModel):
    status: str
    message: str


class DeletedResponse(BaseModel):
    status: str
    documents_deleted: int
