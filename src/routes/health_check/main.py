from fastapi import APIRouter, HTTPException


router = APIRouter()


@router.get("/health-check")
def health_check():
    return {"status": "I'm Alive!!!"}
