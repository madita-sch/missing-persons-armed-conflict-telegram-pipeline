# Import libraries
import pandas as pd
from src.evaluation.evaluation_cv import *
import ast

# Load datasets
gt = pd.read_csv("data/gold_cv.csv")
pred = pd.read_csv("outputs/cv_results.csv")

# Merge on file name
df = pd.merge(gt, pred, on="file")

# Compute color distance
df["gt_rgb"] = df["correct_shirt_color_hex"].apply(hex_to_rgb)
df["pred_rgb"] = df["shirt_color_rgb"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df = compute_color_distance(df)

# Save results
df.to_csv("outputs/evaluation_cv.csv", index=False)

# Evaluation of similarity matrix 
sim_df = pd.read_csv("outputs/image_similarity.csv")
gt_pairs = pd.read_csv("data/gold_cv_image_similarity.csv")

sim_acc, sim_eval_df = evaluate_similarity(sim_df, gt_pairs)

print("Similarity accuracy:", sim_acc)
sim_eval_df.to_csv("outputs/evaluation_similarity.csv", index=False)