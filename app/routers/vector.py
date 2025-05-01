from typing import Annotated

from fastapi import HTTPException, UploadFile, File, APIRouter
import uuid

from app.schemas.vector import *
from app.services.vector_db import Document, WeaviateVectorStorage
from app.services.models import (
    extract_text_from_csv,
    extract_text_from_docx,
    extract_text_from_pdf,
)

router = APIRouter()


class DBSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = WeaviateVectorStorage()
        return cls._instance


@router.post("/file/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        db = DBSingleton()
        contents = await file.read()
        ext = file.filename.lower().split(".")[-1]

        # 1. PDF
        if ext == "pdf":
            text = extract_text_from_pdf(contents)

        # 2. CSV
        elif ext == "csv":
            text = extract_text_from_csv(contents)

        # 3. DOCX
        elif ext == "docx":
            text = extract_text_from_docx(contents)

        else:
            raise HTTPException(status_code=400, detail="Непідтримуваний тип файлу")

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Неможливо прочитати файл: {str(e)}"
        )

    metadata = {"filename": file.filename, "source": "upload", "extension": ext}

    document = Document(content=text, metadata=metadata)

    try:
        db.write_document(document)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "OK", "filename": file.filename}


@router.post("/files/batch")
async def upload_file(files: Annotated[list[UploadFile], File()]):
    try:
        db = DBSingleton()
        documents = []
        for file in files:
            contents = await file.read()
            ext = file.filename.lower().split(".")[-1]
            # 1. PDF
            if ext == "pdf":
                text = extract_text_from_pdf(contents)
            # 2. CSV
            elif ext == "csv":
                text = extract_text_from_csv(contents)
            # 3. DOCX
            elif ext == "docx":
                text = extract_text_from_docx(contents)
            else:
                raise HTTPException(status_code=400, detail="Непідтримуваний тип файлу")

            metadata = {"filename": file.filename, "source": "upload", "extension": ext}

            documents.append(Document(content=text, metadata=metadata))

        try:
            db.batch_write_documents(documents)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        return {"uploaded_files": len(files)}
    except Exception:
        raise


@router.post("/documents", response_model=DocumentIdResponse)
async def write_document(request: DocumentWriteRequest):
    """Store a document with its embedding in vector storage"""
    try:
        db = DBSingleton()

        # Generate ID if not provided
        document = Document(
            id=request.id, content=request.content, metadata=request.metadata
        )

        result_id = db.write_document(document)
        return DocumentIdResponse(document_id=result_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/batch", response_model=DocumentsWrittenResponse)
async def batch_write_documents(request: BatchWriteRequest):
    """Store multiple documents in batch"""
    try:
        db = DBSingleton()

        documents = [
            Document(id=doc.id, content=doc.content, metadata=doc.metadata)
            for doc in request.documents
        ]

        written = db.batch_write_documents(documents)
        return DocumentsWrittenResponse(documents_written=written)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentResponse)
async def get_document():
    """Get a document by ID"""
    try:
        db = DBSingleton()
        document = db.list_documents()
        return {"documents": document}
        # raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/documents/{document_id}", response_model=StatusResponse)
async def update_document(document_id: str, request: DocumentUpdateRequest):
    """Update an existing document"""
    try:
        db = DBSingleton()
        db.update_document(
            document_id=document_id, content=request.content, metadata=request.metadata
        )

        return StatusResponse(status="success", message="Document updated")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents", response_model=StatusResponse | DeletedResponse)
async def delete_documents(request: DeleteRequest):
    """Delete documents by IDs or all documents"""
    try:
        db = DBSingleton()
        deleted = db.delete_documents(
            document_ids=request.document_ids, delete_all=request.delete_all
        )

        if deleted == -1:  # All documents deleted
            return StatusResponse(status="success", message="All documents deleted")
        return DeletedResponse(status="success", documents_deleted=deleted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/by-filters", response_model=DeletedResponse)
async def delete_by_filters(request: DeleteByFiltersRequest):
    """Delete documents by filters"""
    try:
        db = DBSingleton()
        deleted = db.delete_documents_by_filters(request.filters)
        return DeletedResponse(status="success", documents_deleted=deleted)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/similar", response_model=DocumentResponse)
async def search_similar(request: SearchRequest):
    """Search for documents similar to the query"""
    try:
        db = DBSingleton()
        documents = db.search_similar(
            query=request.query,
            filters=request.filters,
            limit=request.limit,
            include_embeddings=request.include_embeddings,
        )

        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/hybrid", response_model=DocumentResponse)
async def search_hybrid(request: HybridSearchRequest):
    """Perform hybrid search (vector + keyword)"""
    try:
        db = DBSingleton()
        documents = db.search_hybrid(
            query=request.query,
            filters=request.filters,
            limit=request.limit,
            alpha=request.alpha,
        )

        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
