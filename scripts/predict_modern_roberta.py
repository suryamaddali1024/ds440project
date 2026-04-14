import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INPUT_FILE = "DS440 recent headlines.csv"
OUTPUT_FILE = "roberta_predictions.csv"
MODEL_PATH = "models/roberta_clickbait"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

df = pd.read_csv(INPUT_FILE)

texts = (df["postText_clean"].fillna("") + " </s> " + df["targetDescription"].fillna("")).tolist()

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

df["roberta_pred"] = probs.argmax(axis=1)
df["roberta_conf"] = probs.max(axis=1)

df.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)