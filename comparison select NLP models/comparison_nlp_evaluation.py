all_rows = []

all_rows.append(seq_row)
all_rows.append(ner_row)
all_rows.append(cluster_row)
all_rows.append(translation_row)  

df_final = pd.DataFrame(all_rows)

df_final.to_excel("FINAL_TASK_COMPARISON.xlsx", index=False)

print(df_final)