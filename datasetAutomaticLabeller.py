import os
import pandas as pd
import requests

# Filepaths for your CSV files
input_csv = 'data/cleaned_company_reputation_data_samsung_posts.csv'
output_csv = 'qwen2.5-7b-instruct_cleaned_company_reputation_data_samsung_posts.csv'

# How often to save intermediate progress (e.g., every 10 rows)
CHECKPOINT_INTERVAL = 10

# LMStudio/QWEN API endpoint (based on the curl example)
api_url = "http://192.168.0.177:1234/v1/chat/completions"


def classify_text(text):
    """Classify text for sentiment towards Samsung, relevance, and provide reasoning."""
    prompt_with_examples = """
You are Samsung's social media analysis assistant. Analyze posts about Samsung.
For each post, determine:
1. The sentiment towards Samsung ('positive', 'negative', or 'neutral')
2. Whether the post is relevant to Samsung ('relevant' or 'irrelevant')
3. Provide a one-sentence reasoning for your classification

Your response MUST follow this exact format:
sentiment: [positive/negative/neutral]
isRelevant: [relevant/irrelevant]
reasoning: [One brief sentence explaining your classifications]

Here are some examples:

Example 1:
text:
As the title suggests, tonight I received an unexpected "Find My Mobile" notification. I have never signed up for, or logged into, such a service. Should I be concerned?

Response:
sentiment: neutral
isRelevant: relevant
reasoning: The user is asking about a Samsung service notification without expressing positive or negative opinions.

Example 2:
text:
I've used Android phones all my life, mostly Samsung devices. Seven months ago, I decided to try the iPhone 15 Pro Max. Right off the bat, I can say there's only one thing I truly loved about it: FaceID... and that's about it. I could go on for an hour listing more reasons why for me, Android is better than iOS. Can't wait to switch back - I'll probably grab the Galaxy S25 when it drops.

Response:
sentiment: positive
isRelevant: relevant
reasoning: The user expresses preference for Samsung devices over iPhone and excitement about returning to Samsung with the Galaxy S25.

Example 3:
text:
The moon pictures from Samsung are fake. Samsung's marketing is deceptive. It is adding detail where there is none (in this experiment, it was intentionally removed).

Response:
sentiment: negative
isRelevant: relevant
reasoning: The post criticizes Samsung's camera technology and marketing as deceptive regarding moon photography features.

Example 4:
text:
Intel has claimed that both products are "re-marked" and not genuine. The problem is that they definitely are not re-marked. They also tried to claim that one of them was a tray processor and thereby not subject to retail warranty, which they backtracked on.

Response:
sentiment: neutral
isRelevant: irrelevant
reasoning: The post discusses issues with Intel CPUs and warranty claims with no mention of Samsung products or services.
"""
    payload = {
        "model": "qwen2.5-7b-instruct",
        "messages": [
            {"role": "system", "content": prompt_with_examples},
            {"role": "user", "content": text}
        ],
        "temperature": 0,
        "max_tokens": -1, 
        "stream": False
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        json_data = response.json()
        response_text = json_data['choices'][0]['message']['content'].strip()

        # Parse the response to extract the classifications and reasoning
        sentiment = None
        is_relevant = None
        reasoning = None

        for line in response_text.split('\n'):
            line = line.strip()
            if line.startswith('sentiment:'):
                sentiment = line.split(':', 1)[1].strip().lower()
            elif line.startswith('isRelevant:'):
                is_relevant = line.split(':', 1)[1].strip().lower()
            elif line.startswith('reasoning:'):
                reasoning = line.split(':', 1)[1].strip()

        return {
            'sentiment': sentiment,
            'is_relevant': is_relevant,
            'reasoning': reasoning
        }
    except requests.exceptions.RequestException as e:
        print(f"Error with the API: {e}")
        return None


def process_csv(input_csv, output_csv):
    """Process the CSV, classify new entries, and track changes with checkpointing."""
    # --- Step 1: Load or initialize the DataFrame ---
    if os.path.exists(output_csv):
        # If a checkpoint file already exists, load it
        df = pd.read_csv(output_csv)
        print(f"Loaded existing progress from '{output_csv}'.")
    else:
        # No checkpoint file - read the original data
        df = pd.read_csv(input_csv)
        print(f"Starting new classification from '{input_csv}'.")

    # --- Step 2: Ensure required columns exist ---
    if 'sentiment' not in df.columns:
        df['sentiment'] = None
    if 'is_relevant' not in df.columns:
        df['is_relevant'] = None
    if 'reasoning' not in df.columns:
        df['reasoning'] = None
    if 'Changed' not in df.columns:
        df['Changed'] = False
    if 'Reclassified' not in df.columns:
        df['Reclassified'] = 0

    # --- Step 3: Find how many rows still need classification ---
    rows_to_classify = df[df['Reclassified'] == 0].shape[0]
    print(f"Rows needing classification: {rows_to_classify}")

    if rows_to_classify == 0:
        print("All rows have already been classified. Exiting.")
        return

    reclass_count = 0  # Number of rows classified this run
    total_classified_now = 0  # To track progress

    # --- Step 4: Classify only rows that haven't been classified yet (Reclassified == 0) ---
    for index, row in df.iterrows():
        if row['Reclassified'] == 1:
            continue  # Skip rows already done

        # Use existing sentiment (if any) for change comparison
        original_sentiment = str(row['sentiment']).lower() if pd.notnull(row['sentiment']) else ''
        text = row['text']
        result = classify_text(text)

        if result:
            df.at[index, 'sentiment'] = result['sentiment']
            df.at[index, 'is_relevant'] = result['is_relevant']
            df.at[index, 'reasoning'] = result['reasoning']
            df.at[index, 'Changed'] = (original_sentiment != result['sentiment'])
            df.at[index, 'Reclassified'] = 1
            reclass_count += 1
            total_classified_now += 1

            # Print progress after each classification
            rows_remaining = rows_to_classify - total_classified_now
            print(f"Classified row index {index}. Sentiment: {result['sentiment']} | "
                  f"Changed: {df.at[index, 'Changed']} | Remaining: {rows_remaining}")

            # --- Step 5: Save a checkpoint if needed ---
            if reclass_count % CHECKPOINT_INTERVAL == 0:
                df.to_csv(output_csv, index=False)
                print(f"[Checkpoint] Saved after {reclass_count} classifications in this run.")

    # --- Step 6: Save final updated CSV after processing ---
    df.to_csv(output_csv, index=False)
    print(f"Updated dataset saved to {output_csv}.")
    print(f"Total newly classified rows in this run: {reclass_count}")


if __name__ == "__main__":
    process_csv(input_csv, output_csv)
