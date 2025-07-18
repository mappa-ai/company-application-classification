from typing import Any
from typing import List
import pandas as pd
from fastapi import APIRouter, HTTPException, Header
import numpy as np


def fetch_data(conn: Any, query: str) -> pd.DataFrame:
    """
    Fetch data from the database using the provided query.
    """
    try:
        result = conn.execute(query).fetchall()
        columns: np.ndarray = np.asarray(
            [desc[0] for desc in conn.execute(query).description]
        )
        embedding_status = pd.DataFrame(result, columns=columns)
        return embedding_status
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()
