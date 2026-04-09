"""
Dashboard-style visualization for student performance dataset.

The dashboard includes:
  1. Pass/Fail ratio (pie chart)
  2. Study hours distribution (histogram)
  3. Attendance trends (histogram + KDE overlay)
  4. Impact of parent education on grades (box plot)
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

from generate_data import generate_student_data


PARENT_EDU_ORDER = ["No Education", "High School", "Associate", "Bachelor", "Master"]

PALETTE = {
    "pass": "#4CAF50",
    "fail": "#F44336",
    "study": "#2196F3",
    "attendance": "#FF9800",
    "edu": ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"],
}


def plot_pass_fail_ratio(ax, df):
    """Pie chart showing pass/fail ratio."""
    counts = df["passed"].value_counts().sort_index(ascending=False)
    labels = ["Pass", "Fail"]
    colors = [PALETTE["pass"], PALETTE["fail"]]
    values = [counts.get(1, 0), counts.get(0, 0)]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    ax.set_title("Pass / Fail Ratio", fontsize=13, fontweight="bold", pad=12)


def plot_study_hours_distribution(ax, df):
    """Histogram of weekly study hours."""
    bins = np.arange(0, 13, 1)
    ax.hist(df["study_hours_per_week"], bins=bins, color=PALETTE["study"],
            edgecolor="white", alpha=0.85)
    ax.set_title("Study Hours Distribution\n(hours per week)", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Study Hours per Week", fontsize=11)
    ax.set_ylabel("Number of Students", fontsize=11)
    mean_val = df["study_hours_per_week"].mean()
    ax.axvline(mean_val, color="navy", linestyle="--", linewidth=1.8,
               label=f"Mean = {mean_val:.1f} h")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def plot_attendance_trends(ax, df):
    """Bar chart of student count across attendance buckets."""
    bins = [0, 40, 55, 70, 85, 100]
    labels = ["<40%", "40–55%", "55–70%", "70–85%", "85–100%"]
    df = df.copy()
    df["att_bucket"] = pd.cut(df["attendance_percentage"], bins=bins,
                              labels=labels, include_lowest=True)
    bucket_counts = df["att_bucket"].value_counts().reindex(labels, fill_value=0)

    bars = ax.bar(bucket_counts.index, bucket_counts.values,
                  color=PALETTE["attendance"], edgecolor="white", alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 2,
                str(int(height)), ha="center", va="bottom", fontsize=9)

    ax.set_title("Attendance Trends\n(students per attendance range)", fontsize=13,
                 fontweight="bold")
    ax.set_xlabel("Attendance Range", fontsize=11)
    ax.set_ylabel("Number of Students", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def plot_parent_education_impact(ax, df):
    """Box plot: grade distribution by parent education level."""
    data_by_edu = [
        df.loc[df["parent_education"] == level, "grade"].dropna().values
        for level in PARENT_EDU_ORDER
    ]

    bp = ax.boxplot(
        data_by_edu,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
    )

    for patch, color in zip(bp["boxes"], PALETTE["edu"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(PARENT_EDU_ORDER) + 1))
    ax.set_xticklabels(PARENT_EDU_ORDER, rotation=15, ha="right", fontsize=9)
    ax.set_title("Parent Education Impact on Grades", fontsize=13, fontweight="bold")
    ax.set_xlabel("Parent Education Level", fontsize=11)
    ax.set_ylabel("Student Grade (0–100)", fontsize=11)
    ax.axhline(50, color="red", linestyle="--", linewidth=1.2, label="Pass threshold (50)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def create_dashboard(df, output_path="student_performance_dashboard.png"):
    """Build and save the 2×2 dashboard figure."""
    fig = plt.figure(figsize=(14, 10), facecolor="#f5f5f5")
    fig.suptitle(
        "Student Performance — Exploratory Data Analysis Dashboard",
        fontsize=16, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in (ax2, ax3, ax4):
        ax.set_facecolor("#fafafa")

    plot_pass_fail_ratio(ax1, df)
    plot_study_hours_distribution(ax2, df)
    plot_attendance_trends(ax3, df)
    plot_parent_education_impact(ax4, df)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Dashboard saved to: {output_path}")
    return fig


if __name__ == "__main__":
    df = generate_student_data()
    create_dashboard(df)
