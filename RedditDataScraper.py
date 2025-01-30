# -*- coding: utf-8 -*-
import os
import asyncio
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import asyncpraw
import nest_asyncio

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Reddit API Configuration
def create_reddit_client():
    return asyncpraw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD")
    )

reddit = create_reddit_client()

async def handle_rate_limit(generator, max_retries=3):
    retries = 0
    while True:
        try:
            async for item in generator:
                yield item
            break
        except Exception as e:
            if "RATELIMIT" in str(e).upper() and retries < max_retries:
                retries += 1
                wait = 5 * retries
                print(f"Rate limit hit. Retrying in {wait} seconds...")
                await asyncio.sleep(wait)
            else:
                raise e

def create_data_container():
    return {
        'content': [],
        'category': [],
        'upvotes': [],
        'type': [],
        'date': []
    }

async def process_submission(submission, category, data):
    try:
        post_date = datetime.utcfromtimestamp(submission.created_utc).isoformat()
        content = submission.title or ""
        if submission.selftext:
            content += f"\n{submission.selftext}".strip()
        
        data['content'].append(content)
        data['category'].append(category)
        data['upvotes'].append(submission.score)
        data['type'].append("post")
        data['date'].append(post_date)

        await process_comments(submission, category, data)
    except Exception as e:
        print(f"Error processing submission {submission.id}: {e}")

async def process_comments(submission, category, data):
    try:
        await submission.comments.replace_more(limit=None)
    except Exception as e:
        if "RATELIMIT" in str(e).upper():
            print("Comment expansion rate limit. Sleeping 10 seconds...")
            await asyncio.sleep(10)
            await submission.comments.replace_more(limit=None)
    
    for comment in submission.comments.list():
        if not isinstance(comment, asyncpraw.models.Comment):
            continue
        
        comment_date = datetime.utcfromtimestamp(comment.created_utc).isoformat()
        data['content'].append(comment.body)
        data['category'].append(category)
        data['upvotes'].append(comment.score)
        data['type'].append("comment")
        data['date'].append(comment_date)

async def search_subreddits(search_term="Samsung", limit=50):
    found = set()
    async for sub in handle_rate_limit(reddit.subreddits.search(search_term, limit=limit)):
        found.add(sub.display_name)
    return found

async def search_posts(search_term="Samsung", limit=1000):
    found = set()
    subreddit = await reddit.subreddit("all")
    async for post in handle_rate_limit(subreddit.search(search_term, limit=limit, sort='date')):
        found.add(post.subreddit.display_name)
    return found

async def fetch_subreddit_content(subreddits, search_term="Samsung", posts_per_sub=100):
    data = create_data_container()
    
    for sub_name in subreddits:
        try:
            subreddit = await reddit.subreddit(sub_name)
            print(f"Processing {sub_name}...")
            
            # Search posts with term
            async for submission in handle_rate_limit(subreddit.search(search_term, limit=posts_per_sub)):
                await process_submission(submission, "queried", data)
            
            # Fetch category posts
            for category in ['hot', 'controversial', 'top', 'new', 'rising']:
                method = getattr(subreddit, category)
                async for submission in handle_rate_limit(method(limit=posts_per_sub)):
                    await process_submission(submission, category, data)
                    
        except Exception as e:
            print(f"Error processing {sub_name}: {e}")
    
    return data

async def main():
    # Collect relevant subreddits
    subs_from_search = await search_subreddits()
    subs_from_posts = await search_posts()
    all_subs = subs_from_search.union(subs_from_posts)
    
    # Fetch and process content
    dataset = await fetch_subreddit_content(all_subs)
    
    # Create and save DataFrame
    df = pd.DataFrame(dataset)
    df.to_csv('company_reputation_data.csv', index=False)
    print(f"Dataset saved with {len(df)} entries")

if __name__ == "__main__":
    asyncio.run(main())