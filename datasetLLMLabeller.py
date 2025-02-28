import os
import pandas as pd
from dotenv import load_dotenv
import requests

load_dotenv()
# Filepaths for your CSV files
input_csv = os.getenv('INPUT_PATH')
output_csv = os.getenv('OUTPUT_PATH')

# How often to save intermediate progress (e.g., every 100 rows)
CHECKPOINT_INTERVAL = 100
LLM_ADDRESS = os.getenv('LLM_ADDRESS')

# LMStudio/QWEN API endpoint (based on the curl example)
api_url = f"http://{LLM_ADDRESS}/v1/chat/completions"

def classify_text(subreddit, text, temperature=0):
    """Classify text for sentiment, relevance, and reasoning."""
    prompt_with_examples = """
You are Samsung's social media analysis assistant. Analyze posts about Samsung. 
For each post, determine:
1. The sentiment towards Samsung ('positive', 'negative', 'neutral', or 'mixed')
2. Whether the post is relevant to Samsung ('relevant' or 'irrelevant')
3. Provide a one-sentence reasoning for your classification

You should rate the sentiment and relevance specifically towards Samsung;
if the public sees this post will their opinion of Samsung improve? then it is Positive and Relevant.
If the post is about Samsung but does not express any opinion or sentiment, then it is Neutral.
If the post is about Samsung but expresses a negative opinion or sentiment that may leave a negative light on Samsung, then it is Negative.
If the post contains both positive and negative opinions, it is Mixed.
Is the post's central focus about Samsung? then it is Relevant, otherwise it is Irrelevant.

Your response MUST follow this exact format:
sentiment: [positive/negative/neutral/mixed]
is_relevant: [relevant/irrelevant]
reasoning: [One brief sentence explaining your classifications]

Here are some examples:

Example 1:
subreddit: SamsungHelp
text:
As the title suggests, tonight I received an unexpected "Find My Mobile" notification. I have never signed up for, or logged into, such a service. Should I be concerned?

Response:
sentiment: neutral
is_relevant: irrelevant
reasoning: The user is asking about a notification, that doesn't have anything inherently to dow ith Samsung and doesn't express positive or negative opinions.

Example 2:
subreddit: Android
text:
I've used Android phones all my life, mostly Samsung devices. Seven months ago, I decided to try the iPhone 15 Pro Max. Right off the bat, I can say there's only one thing I truly loved about it: FaceID... and that's about it. I could go on for an hour listing more reasons why for me, Android is better than iOS. Can't wait to switch back - I'll probably grab the Galaxy S25 when it drops.

Response:
sentiment: positive
is_relevant: relevant
reasoning: The user expresses preference for Samsung devices over iPhone and excitement about returning to Samsung with the Galaxy S25.

Example 3:
subreddit: photography
text:
The moon pictures from Samsung are fake. Samsung's marketing is deceptive. It is adding detail where there is none (in this experiment, it was intentionally removed).

Response:
sentiment: negative
is_relevant: relevant
reasoning: The post criticizes Samsung's camera technology and marketing as deceptive regarding moon photography features.

Example 4:
subreddit: Samsung
text:
The new Galaxy S24 has an amazing camera, but the battery life is disappointing.

Response:
sentiment: mixed
is_relevant: relevant
reasoning: The post expresses both positive (camera) and negative (battery life) sentiments towards Samsung.

Example 5:
subreddit: hardware
text:
Rate my setup, I have a Samsung monitor, a Logitech keyboard, and a Razer mouse.

Response:
sentiment: neutral
is_relevant: irrelevant
reasoning: The post simply mentions Samsung as part of a list of hardware items, but does not express any sentiment or opinion about Samsung, i.e. Samsung is not a focus point of this post. thus it is irrelevant.

Remember that a post that just mentions Samsung is not necessarily relevant to Samsung.
"""
    user_content = f"subreddit: {subreddit}\ntext:\n{text}"
    payload = {
        "model": "qwen2.5-7b-instruct",
        "messages": [
            {"role": "system", "content": prompt_with_examples},
            {"role": "user", "content": user_content}
        ],
        "temperature": temperature,
        "max_tokens": -1,
        "stream": False
    }
    try:
        while True:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            json_data = response.json()
            response_text = json_data['choices'][0]['message']['content'].strip()

            sentiment, is_relevant, reasoning = None, None, None

            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('sentiment:'):
                    sentiment = line.split(':', 1)[1].strip().lower()
                elif line.startswith('is_relevant:'):
                    is_relevant = line.split(':', 1)[1].strip().lower()
                elif line.startswith('reasoning:'):
                    reasoning = line.split(':', 1)[1].strip()

            if sentiment in ['positive', 'negative', 'neutral', 'mixed'] and is_relevant in ['relevant', 'irrelevant'] and reasoning:
                return {
                    'sentiment': sentiment,
                    'is_relevant': is_relevant,
                    'reasoning': reasoning
                }
            else:
                print("Invalid response format received, retrying...")
    except requests.exceptions.RequestException as e:
        print(f"Error with the API: {e}")
        return None

def process_csv(input_csv, output_csv):
    """Process the CSV, classify new entries, and track changes with checkpointing."""
    df = pd.read_csv(output_csv) if os.path.exists(output_csv) else pd.read_csv(input_csv)
    print(f"Processing dataset: {input_csv}")
    
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0, 'relevant': 0, 'irrelevant': 0}
    
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
    
    temperature = 0
    failure_count = 0
    
    for index, row in df.iterrows():
        if row['Reclassified'] == 1:
            continue
        
        subreddit = row['subreddit'] if 'subreddit' in row else "Unknown"
        text = row['text']
        result = classify_text(subreddit, text, temperature)
        
        if result:
            df.at[index, 'sentiment'] = result['sentiment']
            df.at[index, 'is_relevant'] = result['is_relevant']
            df.at[index, 'reasoning'] = result['reasoning']
            df.at[index, 'Reclassified'] = 1
            sentiment_counts[result['sentiment']] += 1
            sentiment_counts[result['is_relevant']] += 1
            
            remaining = len(df[df['Reclassified'] == 0])
            print(f"Classified: {index} | positive: {sentiment_counts['positive']} | negative: {sentiment_counts['negative']} | neutral: {sentiment_counts['neutral']} | mixed: {sentiment_counts['mixed']} | Relevant: {sentiment_counts['relevant']} | Irrelevant: {sentiment_counts['irrelevant']} | Remaining: {remaining} | Temperature: {temperature}")
            temperature = 0 # reset temperature after successful classification
            failure_count = 0
        else:
            failure_count +=1
            if failure_count > 2:
                temperature += 0.1
                print(f"Failure count increased, temperature: {temperature}")
                failure_count = 0 #reset failure count after increasing temperature
    
    print(f"Classification Summary: {sentiment_counts}")
    df.to_csv(output_csv, index=False)
    print(f"Updated dataset saved to {output_csv}")

if __name__ == "__main__":
    process_csv(input_csv, output_csv)