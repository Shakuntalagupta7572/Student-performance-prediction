"""
Train a simple ML model on the student performance dataset.

Pipeline:
  - Load / generate data
  - Feature engineering
  - Train a Random Forest classifier (pass/fail prediction)
  - Report accuracy and classification metrics
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from generate_data import generate_student_data


def train_model(df=None, random_seed=42):
    if df is None:
        df = generate_student_data(random_seed=random_seed)

    feature_cols = ["study_hours_per_week", "attendance_percentage", "parent_education"]
    target_col = "passed"

    df = df.copy()
    le = LabelEncoder()
    df["parent_education_enc"] = le.fit_transform(df["parent_education"])

    X = df[["study_hours_per_week", "attendance_percentage", "parent_education_enc"]]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fail", "Pass"]))

    importances = pd.Series(
        clf.feature_importances_,
        index=["study_hours_per_week", "attendance_percentage", "parent_education"],
    ).sort_values(ascending=False)
    print("\nFeature Importances:")
    print(importances.to_string())

    return clf, acc


if __name__ == "__main__":
    train_model()
