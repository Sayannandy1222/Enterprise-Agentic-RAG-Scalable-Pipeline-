from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/metrics")
def metrics(request: Request):
    s = getattr(request.app.state, "retrieval", None)
    return {"retrieval": s.metrics() if s else {}}
