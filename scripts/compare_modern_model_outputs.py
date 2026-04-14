import pandas as pd

rob = pd.read_csv("results/roberta_predictions.csv")
dis = pd.read_csv("results/distilbert_3class_predictions.csv")

merged = rob.merge(dis, on="ID")

merged["agree"] = merged["roberta_pred"] == merged["distilbert_pred"]

print("Agreement:", merged["agree"].mean())

print("\nDistilBERT ambiguous count:")
print((merged["distilbert_pred"] == 1).sum())

merged.to_csv("model_comparison.csv", index=False)

print("Saved: model_comparison.csv")