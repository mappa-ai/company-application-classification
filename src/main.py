from fastapi import FastAPI
from src.routes.predict_interview.main import router as predict_interview_router
from src.routes.health_check.main import router as health_check_router


app: FastAPI = FastAPI()


app.include_router(
    router=predict_interview_router,
    prefix="",
    tags=["Interview Prediction"],
)
app.include_router(
    router=health_check_router,
    prefix="",
    tags=["Health Check"],
)
