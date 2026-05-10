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
HEADER_FILL  = PatternFill("solid", fgColor="2F4F8F")   # dark blue
HEADER_FONT  = Font(bold=True, color="FFFFFF", name="Arial", size=10)
FP_FILL      = PatternFill("solid", fgColor="FFE0E0")   # light red  → False Positive
FN_FILL      = PatternFill("solid", fgColor="FFF3CD")   # light amber → False Negative
LOW_BLEU     = PatternFill("solid", fgColor="FFD0D0")   # red tint   → BLEU < 0.05
MED_BLEU     = PatternFill("solid", fgColor="FFF3CD")   # amber      → 0.05–0.30
OK_FILL      = PatternFill("solid", fgColor="DFF0D8")   # green tint → good
LEAK_FILL    = PatternFill("solid", fgColor="FFE0E0")
CLS_FILL     = PatternFill("solid", fgColor="E8D5F5")   # purple tint

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


# Create main function that builds the error analysis report with multiple sheets for different error types (classification, NER, translation, pseudonymization leaks) and a full debug sheet with all data side-by-side for manual review.
def build_error_analysis_report(df_pred, df_gold, output_path):
    from src.evaluation.evaluation_nlp import split_entities, is_match

    merged = df_pred.merge(df_gold, on="id", suffixes=("_pred", "_gold"))

    # Classification errors: where is_missing_pred ≠ is_missing_gold
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

    # NER errors: where entities in pred vs gold don't match (split by type: names, location, dates, age)
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

    for _, row in merged.iterrows():
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

    # Translation errors: computes BLEU score between text_clean_en_pred vs text_clean_en_gold, flags low-quality translations
    smoother = SmoothingFunction().method1
    trans_rows = []
    for _, row in merged.iterrows():
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

    # Pseudonymization leaks: flags rows where predicted names (names_pred) appear in the text_clean_anon_pred
    # Define regex pattern for phone numbers (simple heuristic: sequences of 8-15 digits)
    phone_pattern = re.compile(r"\b\d{8,15}\b")
    # Define function to check for leaks in a single row
    leak_rows = []

    for _, row in merged.iterrows():
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

    # Create full debug sheet with all relevant columns side-by-side for manual review
    full_cols = [
        "id",
        "is_missing_pred", "is_missing_gold",
        "names_pred",    "names_gold",
        "location_pred", "location_gold",
        "dates_pred",    "dates_gold",
        "age_pred",      "age_gold",
        "text_clean_pred", "text_clean_gold",
        "text_clean_en_pred", "text_clean_en_gold",
    ]
    full = merged[[c for c in full_cols if c in merged.columns]]

    # Export data in one Excel file with separate sheets by type of error
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        cls.to_excel(writer,         sheet_name="classification",    index=False)
        ner.to_excel(writer,         sheet_name="ner_errors",        index=False)
        translation.to_excel(writer, sheet_name="translation",       index=False)
        leak.to_excel(writer,        sheet_name="pseudonymization",  index=False)
        full.to_excel(writer,        sheet_name="full_debug",        index=False)

    _format_excel(output_path)
    print(f"Saved → {output_path}")

    return {
        "classification": cls,
        "ner":            ner,
        "translation":    translation,
        "pseudonymization": leak,
        "full":           full,
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
        "A": 8, "B": 14, "C": 14,
        "D": 28, "E": 28, "F": 28, "G": 28,
        "H": 18, "I": 18, "J": 10, "K": 10,
        "L": 50, "M": 50, "N": 50, "O": 50,
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
    leg.column_dimensions["A"].width = 22
    leg.column_dimensions["B"].width = 55

    legend_rows = [
        ("LEGEND", ""),
        ("", ""),
        ("Sheet",           "Contents"),
        ("classification",  "Records where pred is_missing ≠ gold is_missing"),
        ("ner_errors",      "NER_FN = gold entity not found in pred | NER_FP = pred entity not in gold"),
        ("translation",     "BLEU score comparison (pred vs gold English translation)"),
        ("pseudonymization","Rows where real names or phone numbers leaked through anonymization"),
        ("full_debug",      "Complete side-by-side table of all predictions vs gold"),
        ("", ""),
        ("Color coding",    ""),
        ("🔴 NER_FP",       "False Positive – pred extracted an entity not in gold"),
        ("🟡 NER_FN",       "False Negative – gold entity was missed by pred"),
        ("🔴 BLEU Low",     "BLEU < 5% — translation very poor"),
        ("🟡 BLEU Med",     "BLEU 5–30% — translation partial"),
        ("🟢 BLEU OK",      "BLEU > 30% — translation good"),
        ("🔴 Pseudonym",    "Name or phone number leaked through anonymization"),
    ]

    for r_idx, (col_a, col_b) in enumerate(legend_rows, 1):
        ca = leg.cell(r_idx, 1, col_a)
        cb = leg.cell(r_idx, 2, col_b)
        if r_idx == 1:
            ca.font = Font(bold=True, size=14, name="Arial", color="2F4F8F")
        elif col_a in ("Sheet", "Color coding"):
            for c in (ca, cb):
                c.font = Font(bold=True, name="Arial", size=10)
                c.fill = PatternFill("solid", fgColor="2F4F8F")
                c.font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
        else:
            for c in (ca, cb):
                c.font = Font(name="Arial", size=10)
        leg.row_dimensions[r_idx].height = 18

    wb.save(path)
