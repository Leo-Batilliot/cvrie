"""
utils.py — Image loading and feature extraction utilities for the supervised pipeline.

This module handles:
  - Reading the dataset index CSV
  - Opening and normalizing MRI images with PIL
  - Extracting flat pixel-vector feature representations
  - Visualization helpers for dataset analysis
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def load_class_counts(file_path: str):
    """
    Count the number of images per class from the dataset CSV.

    Iterates the CSV and tallies occurrences of each unique class_name.
    Returns two parallel lists: class names and their respective counts.

    Parameters
    ----------
    file_path : str
        Path to the dataset index CSV (must have a 'class_name' column).

    Returns
    -------
    names : list[str]
        Unique class names in order of first occurrence.
    data : list[int]
        Count of samples for each corresponding class.
    """
    names = []
    data = []
    try:
        with open(file=file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_name = row['class_name']
                if class_name not in names:
                    names.append(class_name)
                    data.append(0)
                idx = names.index(class_name)
                data[idx] += 1
    except Exception as e:
        print(f"Error : {e}")
    return names, data


def load_sample_filepath(file_path: str):
    """
    Collect up to 5 sample filenames per class from the dataset CSV.

    Useful for building a gallery of representative images per class
    without loading the full dataset into memory.

    Parameters
    ----------
    file_path : str
        Path to the dataset index CSV (must have 'class_name' and 'filename' columns).

    Returns
    -------
    names : list[str]
        Unique class names.
    data : list[list[str]]
        For each class, a list of up to 5 filenames.
    """
    names = []
    data = []
    try:
        with open(file=file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                class_name = row['class_name']
                image = row['filename']
                if class_name not in names:
                    names.append(class_name)
                    data.append([image])
                else:
                    index = names.index(class_name)
                    if len(data[index]) < 5:
                        data[index].append(image)
    except Exception as e:
        print(f"Error: {e}")
    return names, data


def get_all_size(file_path: str, directory: str):
    """
    Collect the pixel dimensions of every image in the dataset.

    Opens each image referenced by the CSV and records its width and height.
    Images that cannot be opened are silently skipped.

    Parameters
    ----------
    file_path : str
        Path to the dataset index CSV.
    directory : str
        Root directory of the image dataset (absolute path recommended).

    Returns
    -------
    widths : list[int]
        Width in pixels for each successfully opened image.
    heights : list[int]
        Height in pixels for each successfully opened image.
    """
    widths = []
    heights = []
    try:
        with open(file=file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                split = row["split"]
                class_name = row["class_name"]
                filename = row["filename"]
                path = f"{directory}/{split}/{class_name}/{filename}"
                try:
                    img = Image.open(path)
                    w, h = img.size

                    widths.append(w)
                    heights.append(h)
                except Exception as e:
                    print(f"Error : {e}")

    except Exception as e:
        print(f"Error : {e}")
    return widths, heights


def get_counts_split(file_path: str):
    """
    Count the number of images in the Training and Testing splits.

    Parameters
    ----------
    file_path : str
        Path to the dataset index CSV (must have a 'split' column
        with values "Training" or "Testing").

    Returns
    -------
    count_training : int
    count_testing : int
    """
    count_testing = 0
    count_training = 0
    try:
        with open(file=file_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row["split"]
                if split == "Training":
                    count_training += 1
                if split == "Testing":
                    count_testing += 1
    except Exception as e:
        print(f"Error : {e}")
    return count_training, count_testing


def plot_image_size_distribution(filepath: str, directory: str):
    """
    Plot histograms of image widths and heights across the full dataset.

    Helps verify whether all images share a consistent resolution or
    whether resizing is needed before feature extraction.

    Parameters
    ----------
    filepath : str
        Path to the dataset index CSV.
    directory : str
        Root directory of the image dataset (absolute path recommended).
    """
    widths, heights = get_all_size(filepath, directory)
    print(f"Total images analyzed: {len(widths)}")
    print(f"Min width: {min(widths)}, Max width: {max(widths)}")
    print(f"Min height: {min(heights)}, Max height: {max(heights)}")

    # LOAD PLT
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    plt.hist(widths, bins=30)
    plt.title("Image Width Distribution")
    plt.xlabel("Width")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    plt.hist(heights, bins=30)
    plt.title("Image Height Distribution")
    plt.xlabel("Height")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()


def load_data(csv_path: str, directory: str, target_size: tuple = (224, 224)):
    """
    Load the full dataset as flat pixel-vector arrays ready for sklearn.

    For each image referenced in the CSV:
      1. Open with PIL and convert to RGB — handles grayscale and RGBA MRIs
         uniformly so every sample has exactly 3 channels.
      2. Resize to ``target_size`` pixels — standardizes all images to the same
         resolution regardless of the original size.
      3. Normalize pixel values to [0, 1] by dividing by 255.
      4. Flatten the (H, W, 3) array to a 1-D feature vector.

    Images that fail to open are silently skipped.

    Resolution / speed trade-off
    -----------------------------
    The ``target_size`` parameter directly controls the feature vector length
    and therefore training time. Smaller resolutions are **much** faster:

    +-----------+-----------+----------+-------------------------------+
    | Size      | Features  | Load     | Notes                         |
    +===========+===========+==========+===============================+
    | (224,224) | 150 528   | ~15 min  | Full quality — production run |
    | (64, 64)  |  12 288   | ~5 min   | Good balance quality/speed    |
    | (32, 32)  |   3 072   | ~3 min   | Fast prototyping / fast mode  |
    +-----------+-----------+----------+-------------------------------+

    Use ``FAST_MODE = True`` in the notebook to switch to (32, 32) automatically.

    Parameters
    ----------
    csv_path : str
        Path to the dataset index CSV.
    directory : str
        Root directory containing Training/ and Testing/ subdirectories.
    target_size : tuple of int, default (224, 224)
        (width, height) to resize every image to before flattening.
        Pass (32, 32) or (64, 64) for much faster training.

    Returns
    -------
    X_train : np.ndarray  shape (n_train, target_size[0]*target_size[1]*3)
    y_train : np.ndarray  shape (n_train,)  — string class labels
    X_test  : np.ndarray  shape (n_test,  target_size[0]*target_size[1]*3)
    y_test  : np.ndarray  shape (n_test,)   — string class labels
    """
    X_train, y_train = [], []
    X_test, y_test = [], []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            split = row["split"]
            class_name = row["class_name"]
            filename = row["filename"]
            class_label = row["class_label"]

            path = f"{directory}/{split}/{class_name}/{filename}"

            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    img = img.resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = img_array.flatten()

                    if split == "Training":
                        X_train.append(img_array)
                        y_train.append(class_label)
                    elif split == "Testing":
                        X_test.append(img_array)
                        y_test.append(class_label)

            except Exception as _:
                pass
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def plot_image_formats(filepath: str, directory: str):
    """
    Plot a bar chart of image format distribution (JPEG, PNG, etc.).

    Verifies format consistency across the dataset. A dataset with mixed
    formats may require additional conversion steps before training.

    Parameters
    ----------
    filepath : str
        Path to the dataset index CSV.
    directory : str
        Root directory of the image dataset (absolute path recommended).
    """
    formats = {}

    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            class_name = row["class_name"]
            filename = row["filename"]
            path = f"{directory}/{split}/{class_name}/{filename}"
            try:
                with Image.open(path) as img:
                    fmt = img.format
                    idx = formats.get(fmt, 0)
                    formats[fmt] = idx + 1
            except Exception as _:
                pass

    plt.figure(figsize=(8, 5))
    bars = plt.bar(formats.keys(), formats.values())
    plt.title("Image Format Distribution")
    plt.ylabel("Number of Images")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha="center", va="bottom")

    plt.tight_layout()
    plt.show()
