#!/usr/bin/env python3
"""
Script to retrain the model with current environment to fix compatibility issues
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
from libsql_client import create_client
import libsql
import json
import ast
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings("ignore")


def fetch_data(conn, query):
    """
    Fetch data from the database using the provided query.
    """
    try:
        result = conn.execute(query).fetchall()
        columns: np.ndarray = np.asarray(
            [desc[0] for desc in conn.execute(query).description]
        )
        df = pd.DataFrame(result, columns=columns)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def main():
    print("🔄 Starting model retraining process...")

    # Load environment variables
    load_dotenv()

    print("📡 Connecting to the database...")
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")

    conn = libsql.connect("src/ml/hello.db", sync_url=url, auth_token=auth_token)
    conn.sync()

    # Fetch data
    print("📊 Fetching embedding data...")
    query_embedding = """
    select *
    from embedded_application
    """
    embedding = fetch_data(conn, query_embedding)

    print("📊 Fetching application status data...")
    query_application_status = """
    select * 
    from application_status
    """
    application_status = fetch_data(conn, query_application_status)

    # Merge and prepare data
    print("🔄 Preparing data...")
    embedding_status = pd.merge(
        embedding,
        application_status,
        left_on="application_id",
        right_on="id",
        how="left",
    )
    embedding_status = embedding_status[["company_and_application_embedding", "status"]]
    embedding_status["status"] = embedding_status["status"].apply(
        lambda x: 1 if x in ["ACCEPTED", "SELECTED", "HIRED", "FINISHED"] else 0
    )

    # Parse embeddings
    print("🔄 Parsing embeddings...")
    embedding_status["parsed_embedding"] = embedding_status[
        "company_and_application_embedding"
    ].apply(lambda x: ast.literal_eval(x.replace("\n", "").strip()))

    # Prepare features and target
    X = np.array(embedding_status["parsed_embedding"].tolist())
    y = np.array(embedding_status["status"])

    print(f"📈 Dataset shape: {X.shape}")
    print(f"📈 Class distribution: {np.bincount(y)}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a simpler model without SMOTE to avoid compatibility issues
    print("🤖 Training RandomForest model...")
    model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=10,
        n_estimators=100,
    )

    # Fit the model
    model.fit(X_train, y_train)

    # Evaluate
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the model
    print("💾 Saving model...")
    os.makedirs("src/best_model", exist_ok=True)
    joblib.dump(model, "src/best_model/best_model.pkl")

    print("✅ Model retrained and saved successfully!")
    print(f"🎯 Final accuracy: {accuracy:.4f}")

    # Test loading the new model
    print("🧪 Testing model loading...")
    try:
        loaded_model = joblib.load("src/best_model/best_model.pkl")
        test_prediction = loaded_model.predict_proba(X_test[:1])[0][1]
        print(f"✅ Model loads successfully! Test prediction: {test_prediction:.4f}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")


if __name__ == "__main__":
    main()
