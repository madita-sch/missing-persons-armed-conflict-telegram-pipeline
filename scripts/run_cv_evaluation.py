# Import libraries
import pandas as pd
from src.evaluation.evaluation_cv import *
import ast


gt = pd.read_csv("data/cv_correct_results.csv")
pred = pd.read_csv("outputs/cv_results.csv")

df = pd.merge(gt, pred, on="file")

df["gt_rgb"] = df["correct_shirt_color_hex"].apply(hex_to_rgb)
df["pred_rgb"] = df["shirt_color_rgb"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df = compute_color_distance(df)

df.to_csv("outputs/evaluation_cv.csv", index=False)
