import requests
from bs4 import BeautifulSoup
import re
import time
from typing import List, Optional
from urcm.core.ingest import KnowledgeIngestion

class WebSensor:
    """
    The Sensory Organ for the Internet.
    Fetches, cleans, and ingests live web data into the Resonance Memory.
    """
    
    def __init__(self, ingestion_engine: Optional[KnowledgeIngestion] = None):
        if ingestion_engine:
            self.ingestor = ingestion_engine
        else:
            # Connect to existing brain
            self.ingestor = KnowledgeIngestion(l2_dim=512)
            
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetches and cleans text from a URL."""
        print(f"🌐 WebSensor: Connecting to {url}...")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
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
