"""
source_analysis.py
------------------
Step 1: Analyze where articles come from (source/website) and how source
relates to clickbait tendency.

Uses the Twitter/X oEmbed API to identify which news organization posted
each tweet, then analyzes clickbait rates by source.

Two parts:
  Part 1: Scrape tweet sources via oEmbed API (saves tweet_sources.csv)
  Part 2: Analyze clickbait patterns by source

Requires: merged_with_urls.csv (dataset subset with twitter_url column)

Usage:
    python source_analysis.py
"""

import urllib.request
import json
import time
import os

import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG
# ===========================================================================
INPUT_FILE = "../data/merged_with_urls.csv"
SOURCES_FILE = "../results/tweet_sources.csv"
REQUEST_DELAY = 0.3  # seconds between API requests (rate limiting)


# ===========================================================================
# PART 1: SCRAPE TWEET SOURCES
# ===========================================================================

def scrape_tweet_sources(df):
    """
    Extract tweet IDs from URLs and look up the posting account via
    Twitter's oEmbed API (no authentication required).

    The oEmbed endpoint returns metadata about a tweet including the
    author's display name and profile URL, without needing API keys.

    Example:
        URL: https://twitter.com/i/web/status/608310377143799810
        Tweet ID: 608310377143799810
        oEmbed request: https://publish.twitter.com/oembed?url=https://twitter.com/i/status/608310377143799810
        Response: {"author_name": "Daily Mail", "author_url": "https://twitter.com/DailyMail", ...}
    """
    # Extract tweet IDs from the twitter_url column using regex
    # The URL format is: https://twitter.com/i/web/status/{TWEET_ID}
    df["tweet_id"] = df["twitter_url"].str.extract(r"status/(\d+)")[0]
    tweet_ids = df["tweet_id"].tolist()

    print("=" * 70)
    print("PART 1: SCRAPING TWEET SOURCES")
    print("=" * 70)
    print(f"   Total tweets to scrape: {len(tweet_ids)}")
    print(f"   Request delay: {REQUEST_DELAY}s")
    print(f"   Estimated time: ~{len(tweet_ids) * REQUEST_DELAY / 60:.0f} minutes")
    print()

    results = []
    success = 0
    failed = 0

    for i, tid in enumerate(tweet_ids):
        # Build the oEmbed API URL
        # This is a public endpoint that doesn't require authentication
        oembed_url = (
            f"https://publish.twitter.com/oembed?"
            f"url=https://twitter.com/i/status/{tid}"
        )

        try:
            req = urllib.request.Request(
                oembed_url, headers={"User-Agent": "Mozilla/5.0"}
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())

            results.append({
                "tweet_id": tid,
                "author_name": data.get("author_name", ""),
                "author_url": data.get("author_url", ""),
            })
            success += 1

        except Exception:
            # Tweet may have been deleted or account suspended
            results.append({
                "tweet_id": tid,
                "author_name": "",
                "author_url": "",
            })
            failed += 1

        # Progress update every 100 tweets
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(tweet_ids)}  success={success}  failed={failed}")

        # Rate limiting to avoid being blocked
        time.sleep(REQUEST_DELAY)

    print(f"\n   Done! Success: {success}  Failed: {failed}")

    # Save to CSV for reuse (so we don't have to re-scrape every time)
    results_df = pd.DataFrame(results)
    results_df.to_csv(SOURCES_FILE, index=False)
    print(f"   Saved to {SOURCES_FILE}")

    return results_df


# ===========================================================================
# PART 2: ANALYZE CLICKBAIT PATTERNS BY SOURCE
# ===========================================================================

def analyze_sources(df, sources_df):
    """
    Merge source data with the original dataset and analyze how
    clickbait rates vary by news organization.

    Key questions:
    - Which sources produce the most/least clickbait?
    - Do digital-native outlets (BuzzFeed, HuffPost) differ from
      traditional news (NYT, CNN)?
    - Could source identity be a useful feature for classification?
    """
    print("\n" + "=" * 70)
    print("PART 2: SOURCE ANALYSIS")
    print("=" * 70)

    # Merge source info with original dataset
    df["tweet_id"] = df["twitter_url"].str.extract(r"status/(\d+)")[0]
    sources_df["tweet_id"] = sources_df["tweet_id"].astype(str)
    merged = df.merge(sources_df, on="tweet_id", how="left")

    # Filter to successfully scraped tweets
    has_source = merged["author_name"].fillna("") != ""
    merged_clean = merged[has_source].copy()
    print(f"\n   Samples with source data: {len(merged_clean)}")
    print(f"   Unique sources: {merged_clean['author_name'].nunique()}")

    # --- Clickbait rate by individual source ---
    print(f"\n   --- Clickbait Rate by Source ---")
    print(f"   {'Source':25s}  {'Total':>5s}  {'CB':>4s}  {'NCB':>4s}  {'CB%':>5s}  {'AvgTrMean':>9s}")
    print(f"   {'-'*25}  {'-'*5}  {'-'*4}  {'-'*4}  {'-'*5}  {'-'*9}")

    source_stats = []
    for source in merged_clean["author_name"].value_counts().head(20).index:
        subset = merged_clean[merged_clean["author_name"] == source]
        total = len(subset)
        cb = (subset["truthClass"] == 1).sum()
        ncb = (subset["truthClass"] == 0).sum()
        cb_rate = cb / total
        avg_tm = subset["truthMean"].mean()
        print(f"   {source:25s}  {total:5d}  {cb:4d}  {ncb:4d}  {100*cb_rate:4.0f}%  {avg_tm:9.3f}")
        source_stats.append({
            "source": source, "total": total, "cb": cb,
            "ncb": ncb, "cb_rate": cb_rate, "avg_truthmean": avg_tm,
        })

    stats_df = pd.DataFrame(source_stats).sort_values("avg_truthmean", ascending=False)

    # --- Ranking ---
    print(f"\n   --- Ranked by Clickbait Tendency (avg truthMean) ---")
    print(f"   Most clickbaity:")
    for _, row in stats_df.head(5).iterrows():
        print(f"     {row['source']:25s}  truthMean={row['avg_truthmean']:.3f}  cb_rate={100*row['cb_rate']:.0f}%")
    print(f"   Least clickbaity:")
    for _, row in stats_df.tail(5).iterrows():
        print(f"     {row['source']:25s}  truthMean={row['avg_truthmean']:.3f}  cb_rate={100*row['cb_rate']:.0f}%")

    # --- Category-level analysis ---
    # Group sources into categories for higher-level insights
    categories = {
        "Traditional News": [
            "The New York Times", "The Guardian", "Fox News", "NBC News",
            "CNN", "The Washington Post", "BBC News (UK)", "The Telegraph",
            "ABC News",
        ],
        "Digital Native": [
            "BuzzFeed", "Mashable", "HuffPost", "Bleacher Report",
            "Business Insider",
        ],
        "Tabloid": ["Daily Mail"],
        "Wire/Aggregator": ["Yahoo", "The Wall Street Journal", "ESPN"],
    }

    print(f"\n   --- Source Category Analysis ---")
    print(f"   {'Category':20s}  {'n':>5s}  {'CB%':>5s}  {'AvgTrMean':>9s}")
    print(f"   {'-'*20}  {'-'*5}  {'-'*5}  {'-'*9}")

    for cat_name, cat_sources in categories.items():
        subset = merged_clean[merged_clean["author_name"].isin(cat_sources)]
        if len(subset) == 0:
            continue
        cb_rate = (subset["truthClass"] == 1).mean()
        avg_tm = subset["truthMean"].mean()
        print(f"   {cat_name:20s}  {len(subset):5d}  {100*cb_rate:4.0f}%  {avg_tm:9.3f}")

    # --- Key insight ---
    trad = merged_clean[merged_clean["author_name"].isin(categories["Traditional News"])]
    digital = merged_clean[merged_clean["author_name"].isin(categories["Digital Native"])]
    trad_cb = (trad["truthClass"] == 1).mean()
    digital_cb = (digital["truthClass"] == 1).mean()

    print(f"\n   --- Key Finding ---")
    print(f"   Digital native outlets produce {digital_cb/trad_cb:.1f}x more clickbait than traditional news")
    print(f"   Traditional news: {100*trad_cb:.0f}% clickbait rate")
    print(f"   Digital native:   {100*digital_cb:.0f}% clickbait rate")

    return stats_df


# ===========================================================================
# RUN
# ===========================================================================

# Load the dataset with URLs
df = pd.read_csv(INPUT_FILE, encoding="latin-1")
print(f"Loaded {len(df)} samples from {INPUT_FILE}\n")

# Part 1: Scrape sources (or load cached results)
if os.path.exists(SOURCES_FILE):
    print(f"Found cached {SOURCES_FILE}, loading instead of re-scraping...")
    sources_df = pd.read_csv(SOURCES_FILE)
    print(f"Loaded {len(sources_df)} source records\n")
else:
    sources_df = scrape_tweet_sources(df)

# Part 2: Analyze
stats_df = analyze_sources(df, sources_df)

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
