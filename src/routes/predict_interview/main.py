from typing import Annotated, List
from fastapi import APIRouter, HTTPException, Header
import pandas as pd
from ...utils.classes.main import Application
from pydantic import BaseModel
import os
import libsql
from ...utils.functions.main import fetch_data
from libsql_client import create_client
import libsql
import json

router = APIRouter()


class PredictBody(BaseModel):
    application_id: int
    threshold: float = 0.64


@router.post("/predict")
def predict(
    body: PredictBody, authorization: Annotated[str, Header(alias="Authorization")]
) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token: str = authorization.split(" ")[1]
    if token != os.getenv("PREDICTOR_API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid token")

    with open("src/utils/sql/embedding_request.sql", "r") as file:
        sql_query: str = file.read()
    sql_query = sql_query.replace("?", str(body.application_id))

    url: str = os.getenv("TURSO_DATABASE_URL", "")
    auth_token: str = os.getenv("TURSO_AUTH_TOKEN", "")
    conn = libsql.connect(
        database="company-behavior-mappa", sync_url=url, auth_token=auth_token
    )
    conn.sync()
    embedding_application = fetch_data(conn, sql_query)

    if embedding_application.empty:
        raise HTTPException(
            status_code=404, detail="No data found for the given application_id"
        )
    if not embedding_application["company_and_application_embedding"][0]:
        raise HTTPException(
            status_code=404, detail="No embedding found for the given application_id"
        )
    embedding: List[float] = json.loads(
        str(embedding_application["company_and_application_embedding"][0])
    )

    application = Application(
        application_id=body.application_id,
        embedding=embedding,
        threshold=body.threshold,
    )

    prediction = application.predict()

    return {
        "application_id": body.application_id,
        "prediction": prediction,
        "threshold": body.threshold,
    }
