import os
import asyncio
import re
import logging
from datetime import datetime
import pytz
import pandas as pd
from dotenv import load_dotenv
import asyncpraw
import nest_asyncio

from langdetect import detect, LangDetectException
from asyncprawcore.exceptions import TooManyRequests, Forbidden, NotFound

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

nest_asyncio.apply()
load_dotenv()

USER_TIMEZONE = pytz.timezone("America/New_York")

def create_reddit_client():
    """
    Validates and creates the asyncpraw Reddit client using environment variables.
    Raises ValueError if a required variable is missing.
    """
    
    required_vars = {
        "client_id":os.getenv("REDDIT_CLIENT_ID"),
        "client_secret":os.getenv("REDDIT_CLIENT_SECRET"),
        "user_agent":os.getenv("REDDIT_USER_AGENT"),
        "username":os.getenv("REDDIT_USERNAME"),
        "password":os.getenv("REDDIT_PASSWORD"),
    }
    for var_name, var_value in required_vars.items():
        if not var_value:
            raise ValueError(f"Missing environment variable: {var_name}")
    return asyncpraw.Reddit(**required_vars)

# Instantiate the Reddit client
reddit = create_reddit_client()

def clean_text(text: str) -> str:
    """
    Removes URLs from the text and strips surrounding whitespace.
    """
    text = re.sub(r'http\S+|www\S+', '', text)
    return text.strip()

def is_english(text: str) -> bool:
    """
    Returns True if the language of the text is English.
    Skips detection for texts shorter than 10 characters to avoid errors.
    """
    if len(text.strip()) < 10:
        return False
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

async def handle_rate_limit(generator, max_retries=3):
    """
    Wraps an async generator to handle Reddit rate limits by retrying.
    Explicitly catches TooManyRequests (429) and sleeps before retrying.
    """
    retries = 0
    while True:
        try:
            async for item in generator:
                yield item
            break
        except TooManyRequests as e:
            if retries < max_retries:
                retries += 1
                wait = 5 * retries
                logger.warning(
                    "[handle_rate_limit] TooManyRequests. Retrying in %s seconds... (Attempt %s/%s)",
                    wait, retries, max_retries
                )
                await asyncio.sleep(wait)
            else:
                logger.error("[handle_rate_limit] Exceeded max_retries. Raising exception.")
                raise
        except Exception as e:
            logger.error("[handle_rate_limit] Unknown exception: %s", e)
            raise

async def fetch_with_retry(coro_func, item_desc: str = "unspecified", max_tries=3):
    """
    Calls a single async function (coro_func) and retries if an exception is raised.
    Specifically checks for TooManyRequests to respect Retry-After.
    Logs the final exception if all retries fail and returns None.
    """
    tries = 0
    last_exception = None

    while tries < max_tries:
        try:
            return await coro_func()  # Call the coroutine function to get a new coroutine each time
        except TooManyRequests as e:
            # If Reddit says "rate limit," honor the Retry-After header if possible
            if e.response and "Retry-After" in e.response.headers:
                wait_time = int(e.response.headers["Retry-After"])
            else:
                wait_time = 30

            tries += 1
            last_exception = e
            logger.warning(
                "[fetch_with_retry] 429 TooManyRequests while %s. Retry after %s seconds "
                "(Attempt %s/%s)",
                item_desc, wait_time, tries, max_tries
            )

            # If we haven’t used up our attempts, wait and try again
            if tries < max_tries:
                await asyncio.sleep(wait_time)

        except Exception as ex:
            # For any other error, we also want to do multiple attempts,
            # so we do not immediately raise. We track the exception and move on.
            tries += 1
            last_exception = ex
            logger.exception(
                "[fetch_with_retry] Error while %s (attempt %s/%s):",
                item_desc, tries, max_tries
            )

            # Optionally add a small sleep before retrying (to avoid immediate repeated failures)
            if tries < max_tries:
                await asyncio.sleep(5)

    # If we get here, we tried max_tries times and failed every time.
    logger.error(
        "[fetch_with_retry] Gave up on %s after %s attempts. Last error was: %s",
        item_desc, max_tries, last_exception
    )
    return None

def create_data_container():
    """
    Returns a dictionary for storing post/comment data in a structured way.
    """
    return {
        'text': [],
        'title': [],
        'upvotes': [],
        'type': [],
        'date': [],
        'post_flair': [],
        'user_flair': [],
        'parent_text': [],
        'subreddit': [],
        'category': [],
    }

def localize_timestamp(utc_timestamp: float) -> str:
    """
    Converts a UTC timestamp (float) to a local time string in ISO format.
    """
    utc_dt = datetime.utcfromtimestamp(utc_timestamp)
    local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(USER_TIMEZONE)
    return local_dt.isoformat()

async def process_submission(submission, category, data):
    """
    Loads a submission, checks if it's English, and stores its post information.
    Then calls process_comments to handle all related comments.
    """
    try:
        loaded = await fetch_with_retry(
            submission.load,  # Pass the method without calling it
            f"loading submission {submission.id}"
        )
        if loaded is None:
            logger.warning("[process_submission] Skipping submission %s after repeated failures.", submission.id)
            return

        post_date = localize_timestamp(submission.created_utc)
        post_title = submission.title or ""
        post_body = submission.selftext or ""
        combined_text = f"{post_title}\n{post_body}".strip()
        cleaned_combined = clean_text(combined_text)

        if not is_english(cleaned_combined):
            return

        data['title'].append(clean_text(post_title))
        data['text'].append(clean_text(post_body))
        data['upvotes'].append(submission.score)
        data['type'].append("post")
        data['date'].append(post_date)
        data['post_flair'].append(submission.link_flair_text or None)
        data['user_flair'].append(submission.author_flair_text or None)
        data['parent_text'].append(None)
        data['subreddit'].append(submission.subreddit.display_name)
        data['category'].append(category)

        await process_comments(submission, category, data)

    except Exception as e:
        logger.error("[process_submission] Error processing submission %s: %s", submission.id, e)

async def process_comments(submission, category, data):
    """
    Retrieves and processes all comments for a submission (in English).
    Also attempts to gather parent comment text in an optimized way if available.
    """
    # Expand comments with fetch_with_retry using a lambda to create a new coroutine each time
    await fetch_with_retry(
        lambda: submission.comments.replace_more(limit=None),
        f"expanding comments for submission {submission.id}"
    )

    for comment in submission.comments.list():
        if not hasattr(comment, "body"):
            continue

        comment_text = clean_text(comment.body)
        if not is_english(comment_text):
            continue

        comment_date = localize_timestamp(comment.created_utc)

        # Attempt to fetch parent comment without an extra API call if possible
        parent_text = None
        parent_obj = comment.parent()
        # If parent_obj is a Comment (not Submission), we can read it directly
        if parent_obj and parent_obj.id != comment.id and hasattr(parent_obj, "body"):
            possible_parent = clean_text(parent_obj.body)
            if is_english(possible_parent):
                parent_text = possible_parent
        else:
            # If the parent is a Submission or something else, we skip or optionally fetch it
            # via an API call (comment.parent_id). For large volumes, consider performance tradeoffs.
            pass

        data['title'].append(None)
        data['text'].append(comment_text)
        data['upvotes'].append(comment.score)
        data['type'].append("comment")
        data['date'].append(comment_date)
        data['post_flair'].append(None)  # Only posts have link flair
        data['user_flair'].append(comment.author_flair_text or None)
        data['parent_text'].append(parent_text)
        data['subreddit'].append(submission.subreddit.display_name)
        data['category'].append(category)

async def search_subreddits(search_term="Samsung", limit=50):
    """
    Searches for subreddits related to the given search_term using Reddit's search.
    """
    found = set()
    logger.info("[search_subreddits] Searching subreddits for '%s' (limit=%s)...", search_term, limit)
    async for sub in handle_rate_limit(reddit.subreddits.search(search_term, limit=limit)):
        found.add(sub.display_name)
    return found

async def search_posts(search_term="Samsung", limit=100):
    """
    Searches across r/all for posts mentioning the search_term,
    returning additional subreddits where it’s discussed.
    """
    found = set()
    logger.info("[search_posts] Searching r/all for '%s' (limit=%s)...", search_term, limit)
    subreddit_all = await reddit.subreddit("all")
    async for post in handle_rate_limit(subreddit_all.search(search_term, limit=limit, sort='date')):
        found.add(post.subreddit.display_name)
    return found

async def fetch_subreddit_content(subreddits, search_term="Samsung", posts_per_sub=100):
    """
    Iterates over a list of subreddits and fetches posts/comments for each in multiple categories.
    Retries if a TooManyRequests (429) is encountered, up to max_tries times.
    """
    data = create_data_container()

    for sub_name in subreddits:
        logger.info("\n[fetch_subreddit_content] Processing subreddit: %s", sub_name)
        tries = 0
        max_tries = 3

        while tries < max_tries:
            try:
                subreddit = await reddit.subreddit(sub_name)

                # 1) 'queried' search for the term
                logger.info("[fetch_subreddit_content]  -> Stage: queried for '%s'", search_term)
                async for submission in handle_rate_limit(
                    subreddit.search(search_term, limit=posts_per_sub)
                ):
                    await process_submission(submission, "queried", data)

                # 2) Known categories (hot, controversial, top, new, rising)
                for category in ['hot', 'controversial', 'top', 'new', 'rising']:
                    logger.info("[fetch_subreddit_content]  -> Stage: %s", category)
                    method = getattr(subreddit, category)
                    async for submission in handle_rate_limit(method(limit=posts_per_sub)):
                        await process_submission(submission, category, data)

                logger.info("[fetch_subreddit_content] Finished processing subreddit: %s", sub_name)
                break  # Successfully processed, exit retry loop

            except TooManyRequests as e:
                # Check Retry-After header or wait 30 seconds
                if e.response and "Retry-After" in e.response.headers:
                    wait_time = int(e.response.headers["Retry-After"])
                else:
                    wait_time = 30

                tries += 1
                logger.warning(
                    "[fetch_subreddit_content] 429 TooManyRequests for %s. "
                    "Retry-After: %s seconds. Attempt %s/%s",
                    sub_name, wait_time, tries, max_tries
                )
                await asyncio.sleep(wait_time)

            except Forbidden:
                logger.warning("[fetch_subreddit_content] Subreddit '%s' is private or banned. Skipping.", sub_name)
                break

            except NotFound:
                logger.warning("[fetch_subreddit_content] Subreddit '%s' does not exist (NotFound). Skipping.", sub_name)
                break

            except Exception as e:
                logger.error("[fetch_subreddit_content] Error processing '%s': %s", sub_name, e)
                break  # Stop retrying for this subreddit

    return data

async def main():
    try:
        logger.info("[main] Starting script...")

        logger.info("[main] Gathering subreddits from search...")
        subs_from_search = await search_subreddits()

        logger.info("[main] Gathering subreddits from posts...")
        subs_from_posts = await search_posts()

        # Combine all unique subreddits
        all_subs = subs_from_search.union(subs_from_posts)
        all_subs = list(all_subs)
        logger.info("[main] Found %d unique subreddits in total.", len(all_subs))

        # If no subreddits found, exit early
        if not all_subs:
            logger.info("[main] No subreddits found. Exiting.")
            return

        logger.info("[main] Starting data fetch...")
        dataset = await fetch_subreddit_content(all_subs)

        logger.info("[main] Data fetch complete. Building DataFrame...")
        df = pd.DataFrame(dataset)

        output_file = 'company_reputation_data.csv'
        logger.info("[main] Saving DataFrame to %s...", output_file)
        df.to_csv(output_file, index=False)
        logger.info("[main] Dataset saved with %d entries. Done!", len(df))
    except KeyboardInterrupt:
        await reddit.close()
        logger.info("[main] Keyboard interrupt detected. Exiting.")

if __name__ == "__main__":
    asyncio.run(main())
