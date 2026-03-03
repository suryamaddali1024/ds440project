import pandas as pd

def clean_and_reorder_dataset(input_file='combined_train.csv', output_file='final_cleaned_full.csv'):
    print(f"1. Loading {input_file}...")
    df = pd.read_csv(input_file)

    # Reorder columns
    print("2. Reordering columns...")
    desired_order = [
        'id',
        'postTimestamp',
        'postMedia',
        'postText',
        'targetCaptions',
        'targetTitle',
        'targetDescription',
        'targetParagraphs',
        'targetKeywords',
        'truthJudgments',
        'truthMean',
        'truthMedian',
        'truthMode',
        'truthClass'
    ]

    # Filter to ensure we only select columns that actually exist
    final_cols = [col for col in desired_order if col in df.columns]
    df = df[final_cols]

    # Encode truthClass: clickbait=1, no-clickbait=0
    print("3. Encoding truthClass...")
    df['truthClass'] = df['truthClass'].map({'clickbait': 1, 'no-clickbait': 0})

    # Save
    print(f"4. Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Done! Shape: {df.shape}")

if __name__ == "__main__":
    clean_and_reorder_dataset()
