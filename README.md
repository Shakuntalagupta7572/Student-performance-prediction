# Student Performance Prediction

ML mini project on student performance prediction.

A dashboard-style visualization was created to provide a complete summary of the dataset.
The graph includes pass/fail ratio, study hour distribution, attendance trends, and the
impact of parent education on grades. This combined visualization helps in understanding
the major factors influencing student performance before machine learning model training.

## Project Structure

| File | Description |
|------|-------------|
| `generate_data.py` | Generates a synthetic student performance dataset (500 students) |
| `dashboard.py` | Creates a 2×2 EDA dashboard and saves it as `student_performance_dashboard.png` |
| `train_model.py` | Trains a Random Forest classifier to predict pass/fail |
| `requirements.txt` | Python package dependencies |

## Dashboard

The dashboard (`student_performance_dashboard.png`) contains four panels:

1. **Pass / Fail Ratio** — pie chart showing overall pass and fail percentages.
2. **Study Hours Distribution** — histogram of weekly study hours with mean marker.
3. **Attendance Trends** — bar chart of student counts across attendance buckets.
4. **Parent Education Impact on Grades** — box plots of grade distribution for each
   parent education level.

## Quick Start

```bash
pip install -r requirements.txt

# Generate dataset
python generate_data.py

# Create EDA dashboard (saves student_performance_dashboard.png)
python dashboard.py

# Train and evaluate the Random Forest model
python train_model.py
```

