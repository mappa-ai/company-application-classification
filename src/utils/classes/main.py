from typing import Any
from sklearn.ensemble._forest import RandomForestClassifier
from typing import List
from fastapi import HTTPException
import joblib
import os
import numpy as np


class Application:
    def __init__(
        self,
        application_id: int,
        embedding: List[float],
        threshold: float = 0.45,
    ) -> None:
        self.application_id = application_id
        self.embedding = embedding
        self.threshold = threshold
        self._model = None

    def _load_model(self):
        """Load model with proper error handling"""
        if self._model is not None:
            return self._model

        try:
            model_path = "src/best_model/best_model.pkl"
            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=500, detail=f"Model file not found at {model_path}"
                )

            # Try to load without triggering imblearn imports during prediction
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded_model = joblib.load(model_path)

            # Cache the model
            self._model = loaded_model
            return loaded_model

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    def predict(self) -> str:
        model: Any = self._load_model()

        # Validate inputs first
        if not all(isinstance(x, (int, float)) for x in self.embedding):
            raise HTTPException(
                status_code=400, detail="Embedding must contain only numeric values"
            )
        if not (0 <= self.threshold <= 1):
            raise HTTPException(
                status_code=400, detail="Threshold must be between 0 and 1"
            )

        try:
            # Convert to numpy array for prediction
            embedding_array = np.array(self.embedding).reshape(1, -1)

            # Make prediction using the loaded model
            if hasattr(model, "predict_proba"):
                # Model has predict_proba method
                prediction_proba = model.predict_proba(embedding_array)[0][1]
            elif hasattr(model, "named_steps"):
                # It's a pipeline, try to get the classifier
                if "clf" in model.named_steps:
                    # Skip SMOTE step for prediction (SMOTE is only for training)
                    classifier = model.named_steps["clf"]
                    prediction_proba = classifier.predict_proba(embedding_array)[0][1]
                else:
                    raise HTTPException(
                        status_code=500, detail="Cannot find classifier in pipeline"
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="Model doesn't support probability prediction",
                )

            # Apply threshold
            if prediction_proba < self.threshold:
                return "REJECTED"
            else:
                return "HIRED"

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    def _predict_proba(self, model, embedding: List[float]) -> float:
        """Legacy method - kept for compatibility"""
        try:
            proba: float = model.predict_proba([embedding])[0][1]
            return proba
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
