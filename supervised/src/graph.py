"""
graph.py — Visualization utilities for the supervised learning pipeline.

All functions in this module produce matplotlib figures for exploratory
data analysis. Pass absolute paths to avoid working-directory dependency.
"""

import matplotlib.pyplot as plt
import csv
from pathlib import Path

from PIL import Image


def plot_class_distribution(names: list[str], data: list[int]):
    """
    Plot a bar chart showing the absolute sample count per class.

    Parameters
    ----------
    names : list[str]
        Class labels (e.g. ["glioma", "meningioma", "notumor", "pituitary"]).
    data : list[int]
        Sample count for each class, aligned with `names`.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [f"C{i}" for i in range(len(names))]
    bars = ax.bar(names, data, color=colors)

    ax.set_ylabel('Number of samples')
    ax.set_title('Class Distribution in Dataset — Sample Count')

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate each bar with its exact count
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


def plot_class_distribution_percent(names: list[str], data: list[int]):
    """
    Plot a bar chart showing each class's share of the total dataset (%).

    Useful for quickly spotting class imbalance that could bias a model
    toward the majority class.

    Parameters
    ----------
    names : list[str]
        Class labels.
    data : list[int]
        Sample count for each class, aligned with `names`.
    """
    # Calculate info
    total = sum(data)
    percentages = [(count / total) * 100 for count in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [f"C{i}" for i in range(len(names))]
    bars = ax.bar(names, percentages, color=colors)

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Class Distribution in Dataset — Share (%)")

    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate each bar with its percentage value
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{pct:.1f}%", ha="center", va="bottom")

    ax.yaxis.set_major_formatter('{x:.0f}')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_corrupted_images(filepath: str):
    """
    Check dataset integrity and plot the valid vs. corrupted image count.

    Attempts to open and verify every image referenced in the CSV using
    PIL's verify() method, which detects truncated or malformed files
    without fully decoding pixel data.

    Image paths are derived from the CSV location: the CSV's parent
    directory is assumed to be the dataset root (Training/ and Testing/
    subdirectories live there).

    Parameters
    ----------
    filepath : str
        Path to the dataset index CSV.
    """
    corrupted = 0
    total = 0

    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            split = row["split"]
            class_name = row["class_name"]
            filename = row["filename"]
            path = str(Path(filepath).parent / split / class_name / filename)
            total += 1

            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as _:
                corrupted += 1

    valid = total - corrupted

    print(f"Total images: {total}")
    print(f"Valid images: {valid}")
    print(f"Corrupted images: {corrupted}")

    # Set values
    labels = ["Valid", "Corrupted"]
    values = [valid, corrupted]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values)

    plt.title("Dataset Integrity Check")
    plt.ylabel("Number of Images")

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")

    plt.show()
