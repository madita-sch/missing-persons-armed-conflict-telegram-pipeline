import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def hex_to_rgb(hex_color):
    if hex_color is None or hex_color == "" or pd.isna(hex_color):
        return None
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_distance(c1, c2):
    if c1 is None or c2 is None:
        return np.nan

    return np.linalg.norm(np.array(c1) - np.array(c2))


def compute_color_distance(df):
    df["color_distance"] = df.apply(
        lambda row: rgb_distance(row["pred_rgb"], row["gt_rgb"]),
        axis=1
    )
    return df


def compute_face_accuracy(df):
    df["face_correct"] = df["face_detected"] == df["correct_face_detection"]
    return df, df["face_correct"].mean()


def evaluate_names(df):
    valid = df[df["gt_name"].notna() & df["pred_name"].notna()]

    report = classification_report(
        valid["gt_name"],
        valid["pred_name"],
        zero_division=0
    )

    cm = confusion_matrix(valid["gt_name"], valid["pred_name"])

    return report, cm