from fastapi import APIRouter, UploadFile, File, HTTPException, Request
import tempfile, os
from app.services.ingestion import IngestionService

router = APIRouter(prefix="/api/v1/documents")


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await file.read())
        path = f.name
    try:
        chunks = IngestionService().ingest(path)
        retrieval = getattr(request.app.state, "retrieval", None)
        if retrieval is not None:
            retrieval.vector_store.upsert(chunks)
            retrieval.bm25.add(chunks)
        return {"filename": file.filename, "chunks": [c.__dict__ for c in chunks]}
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(path)


@router.get("/count")
def count(request: Request):
    retrieval = getattr(request.app.state, "retrieval", None)
    return {"documents": retrieval.vector_store.count() if retrieval else 0}
