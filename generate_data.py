"""
Generate synthetic student performance dataset for visualization and ML training.
"""

import numpy as np
import pandas as pd

def generate_student_data(n_students=500, random_seed=42):
    rng = np.random.default_rng(random_seed)

    parent_education_levels = ["No Education", "High School", "Associate", "Bachelor", "Master"]
    parent_edu = rng.choice(parent_education_levels, size=n_students,
                            p=[0.10, 0.30, 0.20, 0.25, 0.15])

    edu_grade_boost = {
        "No Education": 0,
        "High School": 5,
        "Associate": 10,
        "Bachelor": 15,
        "Master": 20,
    }

    base_grade = rng.normal(loc=60, scale=15, size=n_students)
    edu_boost = np.array([edu_grade_boost[e] for e in parent_edu])

    study_hours = np.clip(rng.normal(loc=5, scale=2, size=n_students), 0, 12)
    attendance = np.clip(rng.normal(loc=75, scale=15, size=n_students), 0, 100)

    grade = np.clip(
        base_grade + edu_boost + study_hours * 2.0 + (attendance - 75) * 0.3,
        0, 100
    )

    passed = (grade >= 50).astype(int)

    df = pd.DataFrame({
        "student_id": np.arange(1, n_students + 1),
        "study_hours_per_week": np.round(study_hours, 1),
        "attendance_percentage": np.round(attendance, 1),
        "parent_education": parent_edu,
        "grade": np.round(grade, 1),
        "passed": passed,
    })

    return df


if __name__ == "__main__":
    df = generate_student_data()
    df.to_csv("student_performance.csv", index=False)
    print(f"Dataset saved: {len(df)} records")
    print(df.head())
