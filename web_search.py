import os
import requests
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

class WebSearchService:
    """
    Web search service using Serper API for real-time web data retrieval.
    """
    
    def __init__(self):
        self.api_key = os.getenv("SERPER_API_KEY")
        if not self.api_key:
            print("[WARNING] SERPER_API_KEY not found in .env file. Web search will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
            print(f"✓ Web Search Service initialized with Serper API (Key: {self.api_key[:10]}...)")

        
        if self.enabled:
            self.base_url = "https://google.serper.dev/search"
            self.headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
        else:
            self.base_url = None
            self.headers = None
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform web search using Serper API.
        
        Args:
            query: Search query string
            num_results: Number of results to return (max 10)
            
        Returns:
            List of search results with title, snippet, and link
        """
        if not self.enabled:
            return []
        
        try:
            payload = {
                "q": query,
                "num": min(num_results, 10)
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Extract organic search results
                if 'organic' in data:
                    for item in data['organic'][:num_results]:
                        results.append({
                            'title': item.get('title', ''),
                            'snippet': item.get('snippet', ''),
                            'link': item.get('link', ''),
                            'source': 'web_search'
                        })
                
                return results
            else:
                print(f"[ERROR] Serper API request failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[ERROR] Web search failed: {e}")
            return []
    
    def search_local_businesses(self, query: str, location: str = "") -> List[Dict[str, Any]]:
        """
        Search for local businesses (like food courts, restaurants).
        
        Args:
            query: Business type query (e.g., "food courts", "restaurants")
            location: Location to search in (e.g., "Italy", "Rome")
            
        Returns:
            List of local business results
        """
        if not self.enabled:
            return []
        
        search_query = f"{query} in {location}" if location else query
        
        try:
            payload = {
                "q": search_query,
                "type": "search",
                "num": 8
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Check for places/local results first
                if 'places' in data:
                    for place in data['places'][:5]:
                        results.append({
                            'name': place.get('title', ''),
                            'address': place.get('address', ''),
                            'rating': place.get('rating', ''),
                            'type': 'local_business',
                            'source': 'web_search'
                        })
                
                # Fallback to organic results
                if not results and 'organic' in data:
                    for item in data['organic'][:5]:
                        results.append({
                            'title': item.get('title', ''),
                            'snippet': item.get('snippet', ''),
                            'link': item.get('link', ''),
                            'type': 'web_result',
                            'source': 'web_search'
                        })
                
                return results
            else:
                print(f"[ERROR] Local search failed: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"[ERROR] Local business search failed: {e}")
            return []
    
    def get_real_time_info(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time information about a specific topic.
        
        Args:
            topic: Topic to search for current information
            
        Returns:
            Dictionary with real-time information or None
        """
        if not self.enabled:
            return None
        
        # Add time-sensitive keywords to get recent information
        query = f"{topic} latest news today current"
        
        results = self.search(query, num_results=3)
        
        if results:
            return {
                'topic': topic,
                'results': results,
                'timestamp': self._get_current_timestamp(),
                'source': 'real_time_search'
            }
        
        return None
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp for real-time data."""
        from datetime import datetime
        return datetime.now().isoformat()