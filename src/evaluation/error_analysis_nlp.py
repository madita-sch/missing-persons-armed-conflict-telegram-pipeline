# Import libraries
import re
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Style helpers to ensure formatting across all sheets to better read errors in final output
HEADER_FILL    = PatternFill("solid", fgColor="2F4F8F")   # dark blue
HEADER_FONT    = Font(bold=True, color="FFFFFF", name="Arial", size=10)
FP_FILL        = PatternFill("solid", fgColor="FFE0E0")   # light red  → False Positive
FN_FILL        = PatternFill("solid", fgColor="FFF3CD")   # light amber → False Negative
LOW_BLEU       = PatternFill("solid", fgColor="FFD0D0")   # red tint   → BLEU < 0.05
MED_BLEU       = PatternFill("solid", fgColor="FFF3CD")   # amber      → 0.05–0.30
OK_FILL        = PatternFill("solid", fgColor="DFF0D8")   # green tint → good
LEAK_FILL      = PatternFill("solid", fgColor="FFE0E0")
CLS_FILL       = PatternFill("solid", fgColor="E8D5F5")   # purple tint
CLUSTER_FILL   = PatternFill("solid", fgColor="D6EAF8")   # light blue → clustering errors

THIN_BORDER  = Border(
    bottom=Side(style="thin", color="CCCCCC")
)

# Define helper functions for Excel formatting
def _header_row(ws, cols):
    ws.append(cols)
    for cell in ws[ws.max_row]:
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[ws.max_row].height = 30


def _set_col_widths(ws, widths: dict):
    """widths = {col_letter: width}"""
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


def _freeze(ws, cell="B2"):
    ws.freeze_panes = cell


def _style_rows(ws, start_row, fill_map_col=None, fill_map=None, default_fill=None):
    alt_fill = PatternFill("solid", fgColor="F7F9FC")
    for i, row in enumerate(ws.iter_rows(min_row=start_row, max_row=ws.max_row)):
        row_fill = alt_fill if i % 2 == 0 else None
        if fill_map_col and fill_map:
            val = row[fill_map_col - 1].value
            if val in fill_map:
                row_fill = fill_map[val]
        for cell in row:
            if row_fill:
                cell.fill = row_fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = THIN_BORDER
            cell.font = Font(name="Arial", size=9)


# Create main function that builds the error analysis report with multiple sheets
# for different error types (classification, NER, translation, clustering,
# pseudonymization leaks) and a full debug sheet with all data side-by-side.
#
# FILTERING LOGIC:
# - classification:    all rows (no filter — this IS the classification step)
# - NER:               gold=1 AND pred=1 only
# - translation:       gold=1 AND pred=1 only
# - clustering_errors: gold=1 AND pred=1 only
# - pseudonymization:  gold=1 AND pred=1 only
# - full_debug:        gold=1 AND pred=1 only
def build_error_analysis_report(df_pred, df_gold, output_path):
    from src.evaluation.evaluation_nlp import split_entities, is_match

    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    # ── Downstream filter ────────────────────────────────────────────────────
    # For NER, translation, clustering, and pseudonymization we only evaluate
    # rows where:
    #   • gold is_missing == 1  → message is genuinely a missing persons report
    #   • pred is_missing == 1  → classifier correctly identified it as such
    # This isolates errors that belong to each downstream step rather than
    # inflating error counts with upstream classification mistakes.
    downstream = merged[
        (merged["is_missing_gold"] == 1) &
        (merged["is_missing_pred"] == 1)
    ].copy()

    # ── Classification errors ────────────────────────────────────────────────
    # No filter here — classification errors ARE the comparison between pred and gold.
    cls = merged[merged["is_missing_pred"] != merged["is_missing_gold"]][[
        "id",
        "is_missing_pred", "is_missing_gold",
        "text_clean_pred", "text_clean_gold",
        "text_clean_en_pred", "text_clean_en_gold",
    ]].copy()
    cls["error_type"] = cls.apply(
        lambda r: "FP_missing (pred=1, gold=0)" if r["is_missing_pred"] == 1
        else "FN_missing (pred=0, gold=1)", axis=1
    )

    # ── NER errors ───────────────────────────────────────────────────────────
    # Uses downstream filter: gold=1 AND pred=1.
    # Only flags entity-level mismatches on messages the classifier got right,
    # so errors reflect NER model behaviour, not classification mistakes.
    ner_rows = []
    ctx_cols = [
        "text_clean_pred", "text_clean_gold",
        "text_clean_en_pred", "text_clean_en_gold",
        "names_pred",    "names_gold",
        "location_pred", "location_gold",
        "dates_pred",    "dates_gold",
        "age_pred",      "age_gold",
        "names_en_pred", "names_en_gold",
        "location_en_pred", "location_en_gold",
        "dates_en_pred", "dates_en_gold",
    ]

    for _, row in downstream.iterrows():
        for col in ["names", "location", "dates", "age"]:
            pred = split_entities(row[f"{col}_pred"])
            gold = split_entities(row[f"{col}_gold"])

            for g in gold:
                if not any(is_match(g, p) for p in pred):
                    ner_rows.append({
                        "id": row["id"],
                        "error_type": "NER_FN",
                        "entity_type": col,
                        "missed_value (gold)": g,
                        "pred_had": "; ".join(pred) if pred else "—",
                        **{c: row.get(c, "") for c in ctx_cols},
                    })

            for p in pred:
                if not any(is_match(p, g) for g in gold):
                    ner_rows.append({
                        "id": row["id"],
                        "error_type": "NER_FP",
                        "entity_type": col,
                        "missed_value (gold)": "—",
                        "pred_had": p,
                        **{c: row.get(c, "") for c in ctx_cols},
                    })

    ner = pd.DataFrame(ner_rows)

    # ── Translation errors ───────────────────────────────────────────────────
    # Uses downstream filter: gold=1 AND pred=1.
    # Translation only applies to missing persons messages, and we only want
    # to evaluate translation quality on messages the classifier handled correctly.
    smoother = SmoothingFunction().method1
    trans_rows = []
    for _, row in downstream.iterrows():
        ref  = str(row.get("text_clean_en_gold", "")).split()
        pred = str(row.get("text_clean_en_pred", "")).split()
        if not ref or not pred:
            continue
        bleu = sentence_bleu([ref], pred, smoothing_function=smoother)
        trans_rows.append({
            "id": row["id"],
            "bleu_score":      round(bleu, 4),
            "bleu_pct":        f"{bleu*100:.1f}%",
            "quality_flag":    "🔴 Low"  if bleu < 0.05 else
                               "🟡 Med"  if bleu < 0.30 else "🟢 OK",
            "text_clean":           row.get("text_clean_pred", ""),
            "text_clean_en_pred":   row.get("text_clean_en_pred", ""),
            "text_clean_en_gold":   row.get("text_clean_en_gold", ""),
        })
    translation = pd.DataFrame(trans_rows).sort_values("bleu_score")

    # ── Clustering errors ────────────────────────────────────────────────────
    # Uses downstream filter: gold=1 AND pred=1.
    # Flags rows where predicted cluster_id differs from gold cluster_id.
    # Three error subtypes:
    #   • wrong_cluster   — both pred and gold have a cluster but they disagree
    #                       (messages incorrectly merged or split across clusters)
    #   • missed_cluster  — gold assigns a cluster but pred returns -1 (singleton)
    #                       (model failed to group a message that belongs to a cluster)
    #   • false_cluster   — pred assigns a cluster but gold says -1
    #                       (model incorrectly grouped a standalone message)
    # ── Clustering errors ────────────────────────────────────────────────────
    cluster_rows = []

    # Get all rows with a cluster assignment in gold (not singletons)
    clustered = downstream[downstream["cluster_id_gold"] != -1].copy()

    # For each gold cluster, check whether pred grouped those same messages together
    for gold_cid, group in clustered.groupby("cluster_id_gold"):
        ids_in_gold_cluster = set(group["id"])

        # What pred cluster_id did each of these messages get assigned?
        pred_assignments = group["cluster_id_pred"].value_counts()

        # If all messages in this gold cluster share the same pred cluster_id
        # (even if the number is different), the clustering is correct
        if len(pred_assignments) == 1 and pred_assignments.index[0] != -1:
            continue  # correct — all grouped together under one pred cluster

        # Otherwise flag each message that was split off or lost
        dominant_pred = pred_assignments.index[0]
        for _, row in group.iterrows():
            if row["cluster_id_pred"] != dominant_pred:
                error_type = "split_from_cluster" if row["cluster_id_pred"] != -1 \
                            else "missed_cluster"
                cluster_rows.append({
                    "id":                 row["id"],
                    "cluster_id_gold":    gold_cid,
                    "cluster_id_pred":    row["cluster_id_pred"],
                    "error_type":         error_type,
                    "names_pred":         row.get("names_pred", ""),
                    "names_gold":         row.get("names_gold", ""),
                    "location_pred":      row.get("location_pred", ""),
                    "text_clean_pred":    row.get("text_clean_pred", ""),
                    "text_clean_en_pred": row.get("text_clean_en_pred", ""),
                })

    # Also flag false clusters: pred grouped messages that gold keeps separate
    for pred_cid, group in downstream[downstream["cluster_id_pred"] != -1].groupby("cluster_id_pred"):
        gold_assignments = group["cluster_id_gold"].value_counts()
        if len(gold_assignments) > 1:
            # Pred merged messages from different gold clusters
            for _, row in group.iterrows():
                cluster_rows.append({
                    "id":                 row["id"],
                    "cluster_id_gold":    row["cluster_id_gold"],
                    "cluster_id_pred":    pred_cid,
                    "error_type":         "false_merge",
                    "names_pred":         row.get("names_pred", ""),
                    "names_gold":         row.get("names_gold", ""),
                    "location_pred":      row.get("location_pred", ""),
                    "text_clean_pred":    row.get("text_clean_pred", ""),
                    "text_clean_en_pred": row.get("text_clean_en_pred", ""),
                })

    clustering_errors = pd.DataFrame(cluster_rows).drop_duplicates(subset=["id", "error_type"])

    # ── Pseudonymization leaks ───────────────────────────────────────────────
    # Uses downstream filter: gold=1 AND pred=1.
    # Anonymization is only applied to missing persons messages, and we evaluate
    # leaks only on correctly classified rows to avoid noise from the classifier.
    phone_pattern = re.compile(r"\b\d{8,15}\b")
    leak_rows = []

    for _, row in downstream.iterrows():
        anon_text = str(row.get("text_clean_anon_pred", ""))
        leaks = []

        for name in str(row.get("names_pred", "")).split(";"):
            name = name.strip()
            if name and len(name) > 2 and name in anon_text:
                leaks.append(("name_leak", name))

        for phone in phone_pattern.findall(anon_text):
            leaks.append(("phone_leak", phone))

        if leaks:
            name_leaks  = [v for t, v in leaks if t == "name_leak"]
            phone_leaks = [v for t, v in leaks if t == "phone_leak"]
            leak_rows.append({
                "id": row["id"],
                "leaked_names":    "; ".join(name_leaks)  or "—",
                "leaked_phones":   "; ".join(phone_leaks) or "—",
                "n_name_leaks":    len(name_leaks),
                "n_phone_leaks":   len(phone_leaks),
                "names_pred":           row.get("names_pred", ""),
                "text_clean_pred":      row.get("text_clean_pred", ""),
                "text_clean_anon_pred": anon_text,
                "text_clean_anon_gold": row.get("text_clean_anon_gold", ""),
            })

    leak = pd.DataFrame(leak_rows)

    # ── Full debug sheet ─────────────────────────────────────────────────────
    # Shows all rows side-by-side for manual review.
    # Uses the downstream filter so the debug view stays consistent with the
    # other sheets — only correctly classified missing persons messages.
    # Includes cluster_id columns so clustering mismatches are visible inline.
    full_cols = [
        "id",
        "is_missing_pred", "is_missing_gold",
        "cluster_id_pred", "cluster_id_gold",
        "names_pred",    "names_gold",
        "location_pred", "location_gold",
        "dates_pred",    "dates_gold",
        "age_pred",      "age_gold",
        "text_clean_pred", "text_clean_gold",
        "text_clean_en_pred", "text_clean_en_gold",
    ]
    full = downstream[[c for c in full_cols if c in downstream.columns]]

    # Print row counts so it is clear how many rows each sheet covers
    print(f"  Classification errors : {len(cls)} rows  (all messages, pred ≠ gold)")
    print(f"  NER errors            : {len(ner)} rows  (gold=1 & pred=1 only)")
    print(f"  Translation           : {len(translation)} rows  (gold=1 & pred=1 only)")
    print(f"  Clustering errors     : {len(clustering_errors)} rows  (gold=1 & pred=1 only)")
    print(f"  Pseudonymization      : {len(leak)} rows  (gold=1 & pred=1 only)")
    print(f"  Full debug            : {len(full)} rows  (gold=1 & pred=1 only)")

    # Export data in one Excel file with separate sheets by type of error
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        cls.to_excel(writer,               sheet_name="classification",    index=False)
        ner.to_excel(writer,               sheet_name="ner_errors",        index=False)
        translation.to_excel(writer,       sheet_name="translation",       index=False)
        clustering_errors.to_excel(writer, sheet_name="clustering_errors", index=False)
        leak.to_excel(writer,              sheet_name="pseudonymization",  index=False)
        full.to_excel(writer,              sheet_name="full_debug",        index=False)

    _format_excel(output_path)
    print(f"Saved → {output_path}")

    return {
        "classification":    cls,
        "ner":               ner,
        "translation":       translation,
        "clustering_errors": clustering_errors,
        "pseudonymization":  leak,
        "full":              full,
    }


# Formatting Excel
def _format_excel(path):
    wb = load_workbook(path)

    # Format classification sheet
    ws = wb["classification"]
    _freeze(ws, "C2")
    _set_col_widths(ws, {
        "A": 8, "B": 14, "C": 14, "D": 50, "E": 50, "F": 50, "G": 50, "H": 28,
    })
    for cell in ws[1]:
        cell.font  = HEADER_FONT
        cell.fill  = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for i, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row)):
        for cell in row:
            cell.fill      = CLS_FILL
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.font      = Font(name="Arial", size=9)
            cell.border    = THIN_BORDER
        ws.row_dimensions[row[0].row].height = 60

    # Format NER errors sheet
    ws = wb["ner_errors"]
    _freeze(ws, "E2")
    _set_col_widths(ws, {
        "A": 8,   # id
        "B": 12,  # error_type
        "C": 12,  # entity_type
        "D": 30,  # missed_value
        "E": 30,  # pred_had
        "F": 45,  # text_clean_pred
        "G": 45,  # text_clean_gold
        "H": 45,  # text_clean_en_pred
        "I": 45,  # text_clean_en_gold
        "J": 28,  # names_pred
        "K": 28,  # names_gold
        "L": 28,  # location_pred
        "M": 28,  # location_gold
        "N": 18,  # dates_pred
        "O": 18,  # dates_gold
        "P": 10,  # age_pred
        "Q": 10,  # age_gold
        "R": 28,  # names_en_pred
        "S": 28,  # names_en_gold
        "T": 28,  # location_en_pred
        "U": 28,  # location_en_gold
        "V": 18,  # dates_en_pred
        "W": 18,  # dates_en_gold
    })
    for cell in ws[1]:
        cell.font  = HEADER_FONT
        cell.fill  = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    ner_color = {"NER_FP": FP_FILL, "NER_FN": FN_FILL}
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        etype = row[1].value  # column B = error_type
        fill  = ner_color.get(etype, None)
        for cell in row:
            if fill:
                cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.font      = Font(name="Arial", size=9)
            cell.border    = THIN_BORDER
        ws.row_dimensions[row[0].row].height = 55

    # Format translation sheet
    ws = wb["translation"]
    _freeze(ws, "D2")
    _set_col_widths(ws, {
        "A": 8,   # id
        "B": 10,  # bleu_score
        "C": 10,  # bleu_pct
        "D": 12,  # quality_flag
        "E": 50,  # text_clean
        "F": 50,  # text_clean_en_pred
        "G": 50,  # text_clean_en_gold
    })
    for cell in ws[1]:
        cell.font  = HEADER_FONT
        cell.fill  = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        flag  = str(row[3].value or "")
        fill  = LOW_BLEU if "Low" in flag else MED_BLEU if "Med" in flag else OK_FILL
        for cell in row:
            cell.fill      = fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.font      = Font(name="Arial", size=9)
            cell.border    = THIN_BORDER
        ws.row_dimensions[row[0].row].height = 60

    # Format clustering errors sheet
    if "clustering_errors" in wb.sheetnames:
        ws = wb["clustering_errors"]
        _freeze(ws, "C2")
        _set_col_widths(ws, {
            "A": 8,   # id
            "B": 14,  # cluster_id_pred
            "C": 14,  # cluster_id_gold
            "D": 18,  # error_type
            "E": 28,  # names_pred
            "F": 28,  # names_gold
            "G": 28,  # location_pred
            "H": 28,  # location_gold
            "I": 18,  # dates_pred
            "J": 18,  # dates_gold
            "K": 50,  # text_clean_pred
            "L": 50,  # text_clean_en_pred
        })
        for cell in ws[1]:
            cell.font  = HEADER_FONT
            cell.fill  = HEADER_FILL
            cell.alignment = Alignment(horizontal="center", wrap_text=True)
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.fill      = CLUSTER_FILL
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                cell.font      = Font(name="Arial", size=9)
                cell.border    = THIN_BORDER
            ws.row_dimensions[row[0].row].height = 55

    # Format Pseudonymization sheet
    ws = wb["pseudonymization"]
    _freeze(ws, "C2")
    _set_col_widths(ws, {
        "A": 8,   # id
        "B": 35,  # leaked_names
        "C": 25,  # leaked_phones
        "D": 12,  # n_name_leaks
        "E": 12,  # n_phone_leaks
        "F": 30,  # names_pred
        "G": 50,  # text_clean_pred
        "H": 50,  # text_clean_anon_pred
        "I": 50,  # text_clean_anon_gold
    })
    for cell in ws[1]:
        cell.font  = HEADER_FONT
        cell.fill  = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.fill      = LEAK_FILL
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.font      = Font(name="Arial", size=9)
            cell.border    = THIN_BORDER
        ws.row_dimensions[row[0].row].height = 65

    # Format full debug sheet
    ws = wb["full_debug"]
    _freeze(ws, "D2")
    _set_col_widths(ws, {
        "A": 8,  "B": 14, "C": 14,
        "D": 14, "E": 14,           # cluster_id_pred, cluster_id_gold
        "F": 28, "G": 28, "H": 28, "I": 28,
        "J": 18, "K": 18, "L": 10, "M": 10,
        "N": 50, "O": 50, "P": 50, "Q": 50,
    })
    for cell in ws[1]:
        cell.font  = HEADER_FONT
        cell.fill  = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    alt_fill = PatternFill("solid", fgColor="F4F6FA")
    for i, row in enumerate(ws.iter_rows(min_row=2, max_row=ws.max_row)):
        fill = alt_fill if i % 2 == 0 else None
        for cell in row:
            if fill:
                cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.font      = Font(name="Arial", size=9)
            cell.border    = THIN_BORDER
        ws.row_dimensions[row[0].row].height = 60

    # Create a legend sheet for the error analysis
    leg = wb.create_sheet("LEGEND", 0)
    leg.sheet_view.showGridLines = False
    leg.column_dimensions["A"].width = 28
    leg.column_dimensions["B"].width = 65

    legend_rows = [
        ("LEGEND", ""),
        ("", ""),
        ("Filtering logic", ""),
        ("classification",
         "All rows — no filter. Classification errors ARE the pred vs gold comparison."),
        ("ner_errors",
         "gold=1 AND pred=1 only. Isolates NER errors from classification mistakes."),
        ("translation",
         "gold=1 AND pred=1 only. Evaluates translation quality on correctly classified rows."),
        ("clustering_errors",
         "gold=1 AND pred=1 only. Flags rows where pred cluster_id ≠ gold cluster_id."),
        ("pseudonymization",
         "gold=1 AND pred=1 only. Evaluates anonymization on correctly classified rows."),
        ("full_debug",
         "gold=1 AND pred=1 only. Side-by-side view of all downstream predictions incl. cluster IDs."),
        ("", ""),
        ("Sheet",              "Contents"),
        ("classification",     "Records where pred is_missing ≠ gold is_missing"),
        ("ner_errors",         "NER_FN = gold entity not found in pred | NER_FP = pred entity not in gold"),
        ("translation",        "BLEU score comparison (pred vs gold English translation)"),
        ("clustering_errors",  "wrong_cluster / missed_cluster / false_cluster — see color coding below"),
        ("pseudonymization",   "Rows where real names or phone numbers leaked through anonymization"),
        ("full_debug",         "Complete side-by-side table of all predictions vs gold"),
        ("", ""),
        ("Color coding",       ""),
        ("🔴 NER_FP",          "False Positive – pred extracted an entity not in gold"),
        ("🟡 NER_FN",          "False Negative – gold entity was missed by pred"),
        ("🔴 BLEU Low",        "BLEU < 5% — translation very poor"),
        ("🟡 BLEU Med",        "BLEU 5–30% — translation partial"),
        ("🟢 BLEU OK",         "BLEU > 30% — translation good"),
        ("🔴 Pseudonym",       "Name or phone number leaked through anonymization"),
        ("🔵 wrong_cluster",   "Both pred and gold have a cluster assigned but they disagree"),
        ("🔵 missed_cluster",  "Gold assigns a cluster but pred returned -1 (singleton)"),
        ("🔵 false_cluster",   "Pred assigns a cluster but gold says -1"),
    ]

    for r_idx, (col_a, col_b) in enumerate(legend_rows, 1):
        ca = leg.cell(r_idx, 1, col_a)
        cb = leg.cell(r_idx, 2, col_b)
        if r_idx == 1:
            ca.font = Font(bold=True, size=14, name="Arial", color="2F4F8F")
        elif col_a in ("Sheet", "Color coding", "Filtering logic"):
            for c in (ca, cb):
                c.font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
                c.fill = PatternFill("solid", fgColor="2F4F8F")
        else:
            for c in (ca, cb):
                c.font = Font(name="Arial", size=10)
        leg.row_dimensions[r_idx].height = 18

    wb.save(path)
