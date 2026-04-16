import pandas as pd

# Your dataset
sanjana = pd.read_csv("data/recent headlines.csv")
sanjana_clean = pd.DataFrame({
    "source": sanjana["postText_clean"],
    "headline": sanjana["targetTitle_clean"],
    "description": sanjana["targetDescription"],
    "published": None
})

print(sanjana_clean.head())

# Surya dataset
surya = pd.read_csv("data/modern_headlines.csv")
print(surya.columns)

surya_clean = pd.DataFrame({
    "source": surya["source"],
    "headline": surya["headline"],
    "description": surya["description"],
    "published": surya["published"]
})


# Combine
combined = pd.concat([sanjana_clean, surya_clean], ignore_index=True)

# Save
combined.to_csv("combined_modern_cleaned.csv", index=False)

print(combined.head())
print(combined.shape)
