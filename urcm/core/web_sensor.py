import ipaddress
import re
import time
from typing import List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from urcm.core.ingest import KnowledgeIngestion


class WebSensor:
    """
    The Sensory Organ for the Internet.
    Fetches, cleans, and ingests live web data into the Resonance Memory.
    """

    ALLOWED_SCHEMES = {"http", "https"}
    BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

    def __init__(self, ingestion_engine: Optional[KnowledgeIngestion] = None):
        if ingestion_engine:
            self.ingestor = ingestion_engine
        else:
            # Connect to existing brain
            self.ingestor = KnowledgeIngestion(l2_dim=512)

    def _validate_url(self, url: str) -> bool:
        """Validate URL to prevent SSRF attacks."""
        try:
            parsed = urlparse(url)
        except ValueError:
            return False

        if parsed.scheme not in self.ALLOWED_SCHEMES:
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        if hostname.lower() in self.BLOCKED_HOSTS:
            return False

        # Block private/internal IP ranges
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return False
        except ValueError:
            # hostname is a domain name, not an IP — that's fine
            pass

        return True

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetches and cleans text from a URL."""
        if not self._validate_url(url):
            print(f"❌ WebSensor: Blocked URL (SSRF prevention): {url}")
            return None

        print(f"🌐 WebSensor: Connecting to {url}...")
        try:
            headers = {
                'User-Agent': 'ResonanceAI/0.2.0 (Research Bot)'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"❌ Failed to fetch: Status {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove junk
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            text = soup.get_text()

            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = ' '.join(chunk for chunk in chunks if chunk)

            return clean_text

        except Exception as e:
            print(f"❌ WebSensor Error: {e}")
            return None

    def ingest_url(self, url: str):
        """Reads a page and deposits it into memory."""
        text = self.fetch_page(url)
        if text:
            print(f"📄 Extracted {len(text)} chars. Ingesting...")
            # We ingest only the first 5000 chars to prevent overwhelming the small brain
            self.ingestor.ingest_text(text[:5000])
            self.ingestor.save()
            print("✅ Knowledge deposited.")

    def search_and_learn(self, topic: str, max_results: int = 3):
        """
        Naive search implementation.
        In a real production system, this would use a Search API.
        Here we try to guess a Wikipedia URL or similar.
        """
        print(f"🔍 Seeking knowledge about: {topic}")

        # Heuristic: Try Wikipedia first
        wiki_url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
        self.ingest_url(wiki_url)

        # If we had a search API, we would iterate results here.
        # For now, we simulate "Browsing" by checking if the page existed.

if __name__ == "__main__":
    # Test
    sensor = WebSensor()
    # Test with a stable, knowledge-rich URL
    sensor.ingest_url("https://en.wikipedia.org/wiki/Resonance")
