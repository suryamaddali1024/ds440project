"""
scrape_modern_headlines.py
--------------------------
Scrape current news headlines from RSS feeds for testing the clickbait
classifier on modern data (post-2017, beyond our training distribution).

Pulls headlines from a mix of sources:
  - Traditional news (BBC, NPR, Reuters, Fox News) - mostly non-clickbait
  - Digital native / aggregator (BuzzFeed, HuffPost) - mix of clickbait

Output: CSV with columns headline, description, source, link, published
that can be fed directly into clickbait_inference.py.

Usage:
    python scrape_modern_headlines.py
    python scrape_modern_headlines.py --output ../data/modern_headlines.csv
    python scrape_modern_headlines.py --max_per_feed 50
"""

import argparse
import time

import feedparser
import pandas as pd

# RSS feeds to scrape
# Mix of traditional news and digital native sources for class diversity
FEEDS = {
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "BuzzFeed": "https://www.buzzfeed.com/index.xml",
    "Fox News": "https://moxie.foxnews.com/google-publisher/latest.xml",
    "NPR": "https://feeds.npr.org/1001/rss.xml",
    "The Verge": "https://www.theverge.com/rss/index.xml",
}


def scrape_feed(name, url, max_entries=50):
    """Scrape a single RSS feed and return a list of headline dicts."""
    print(f"   Fetching {name}: {url}")
    try:
        parsed = feedparser.parse(url)
        if parsed.bozo and not parsed.entries:
            print(f"     ERROR: {parsed.bozo_exception}")
            return []

        headlines = []
        for entry in parsed.entries[:max_entries]:
            headlines.append({
                "source": name,
                "headline": entry.get("title", "").strip(),
                "description": entry.get("summary", entry.get("description", "")).strip(),
                "link": entry.get("link", ""),
                "published": entry.get("published", entry.get("updated", "")),
            })
        print(f"     Got {len(headlines)} entries")
        return headlines

    except Exception as e:
        print(f"     ERROR: {type(e).__name__}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Scrape modern news headlines from RSS feeds")
    parser.add_argument("--output", default="../data/modern_headlines.csv",
                        help="Output CSV path")
    parser.add_argument("--max_per_feed", type=int, default=50,
                        help="Max headlines per feed (default: 50)")
    args = parser.parse_args()

    print("=" * 70)
    print("MODERN HEADLINE SCRAPER (RSS feeds)")
    print("=" * 70)
    print(f"   Output: {args.output}")
    print(f"   Max per feed: {args.max_per_feed}")
    print(f"   Feeds: {len(FEEDS)}")
    print()

    all_headlines = []
    for name, url in FEEDS.items():
        headlines = scrape_feed(name, url, args.max_per_feed)
        all_headlines.extend(headlines)
        time.sleep(0.5)  # be polite to the servers

    if not all_headlines:
        print("\n   No headlines scraped. Check network and feed URLs.")
        return

    df = pd.DataFrame(all_headlines)

    # Drop empty headlines
    df = df[df.headline.str.strip() != ""].reset_index(drop=True)

    # Strip HTML tags from descriptions (RSS often includes inline HTML)
    df["description"] = df["description"].str.replace(r"<[^>]+>", "", regex=True).str.strip()

    df.to_csv(args.output, index=False)

    print()
    print(f"=" * 70)
    print(f"SUMMARY")
    print(f"=" * 70)
    print(f"   Total headlines scraped: {len(df)}")
    print()
    print(f"   By source:")
    for source, count in df["source"].value_counts().items():
        print(f"     {source:15s}  {count}")

    print()
    print(f"   Sample headlines:")
    for source in df["source"].unique():
        subset = df[df["source"] == source]
        if len(subset) == 0:
            continue
        sample = subset.iloc[0]
        title = str(sample["headline"])[:80].encode("ascii", "replace").decode()
        print(f"     [{source}] {title}")

    print()
    print(f"   Saved to {args.output}")
    print(f"\n   To classify with our model:")
    print(f"     python clickbait_inference.py --input {args.output} --text_col headline --title_col description")


if __name__ == "__main__":
    main()
