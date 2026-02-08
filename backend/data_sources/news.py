"""
News Sentiment Tool for GOATlens

Optional tool for agents to retrieve recent news sentiment about a company.
This demonstrates "tool selection" - agents decide when to use this tool
based on their analysis needs (e.g., high volatility, recent earnings, etc.).

Uses NewsAPI (free tier: 100 requests/day) or falls back to a simple
sentiment analysis of recent headlines.

This is a learning exercise for Step 6: Tool Calling in AI Product Sense.
"""

import os
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class NewsError(Exception):
    """Exception raised for news API errors."""
    pass


class NewsClient:
    """
    Client for fetching news sentiment about a company.
    
    This is a "tool" that agents can optionally use when they need
    additional context (e.g., recent events, market sentiment).
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize news client.
        
        Args:
            api_key: NewsAPI key (optional - system works without it)
        """
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        self.base_url = "https://newsapi.org/v2"
    
    async def get_sentiment(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        days: int = 7,
    ) -> Dict[str, Any]:
        """
        Get news sentiment for a company.
        
        This is the "tool" that agents can call when they need
        additional context beyond financial data.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name (for better search results)
            days: Number of days to look back
            
        Returns:
            Dictionary with sentiment score, article count, and sample headlines
        """
        if not self.api_key:
            # Graceful degradation: return neutral sentiment if no API key
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "article_count": 0,
                "headlines": [],
                "source": "none",
            }
        
        try:
            # Use company name if available, otherwise ticker
            query = company_name or ticker
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                # NewsAPI free tier: /everything endpoint
                url = f"{self.base_url}/everything"
                params = {
                    "q": query,
                    "apiKey": self.api_key,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "pageSize": 10,  # Free tier limit
                    "from": (datetime.now() - timedelta(days=days)).isoformat(),
                }
                
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                articles = data.get("articles", [])
                
                # Simple sentiment: count positive/negative keywords
                # (In production, you'd use a proper sentiment analysis model)
                positive_keywords = ["beat", "surge", "growth", "gain", "up", "strong", "rise"]
                negative_keywords = ["miss", "drop", "fall", "down", "weak", "decline", "loss"]
                
                sentiment_scores = []
                headlines = []
                
                for article in articles[:5]:  # Top 5 articles
                    title = article.get("title", "").lower()
                    headline = article.get("title", "")
                    
                    pos_count = sum(1 for kw in positive_keywords if kw in title)
                    neg_count = sum(1 for kw in negative_keywords if kw in title)
                    
                    if pos_count > neg_count:
                        sentiment_scores.append(1)
                    elif neg_count > pos_count:
                        sentiment_scores.append(-1)
                    else:
                        sentiment_scores.append(0)
                    
                    headlines.append(headline)
                
                # Calculate average sentiment
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
                
                if avg_sentiment > 0.2:
                    sentiment_label = "positive"
                elif avg_sentiment < -0.2:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                return {
                    "sentiment": sentiment_label,
                    "sentiment_score": round(avg_sentiment, 2),
                    "article_count": len(articles),
                    "headlines": headlines[:3],  # Top 3 for context
                    "source": "newsapi",
                }
        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Rate limit - graceful degradation
                return {
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "article_count": 0,
                    "headlines": [],
                    "source": "rate_limited",
                }
            raise NewsError(f"News API error: {e.response.status_code}")
        except Exception as e:
            # Graceful degradation: return neutral if anything fails
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "article_count": 0,
                "headlines": [],
                "source": "error",
            }
    
    async def should_use_tool(
        self,
        ticker: str,
        financials: Dict[str, Any],
    ) -> bool:
        """
        Determine if news sentiment tool should be used.
        
        This is "tool selection" - agents decide when to use this tool
        based on conditions (e.g., high volatility, recent earnings, etc.).
        
        Args:
            ticker: Stock ticker
            financials: Company financial data
            
        Returns:
            True if tool should be used, False otherwise
        """
        # Example conditions for tool usage:
        # - High volatility (beta > 1.5)
        # - Recent earnings (within last 7 days)
        # - Low analyst coverage (institutional ownership < 30%)
        
        beta = financials.get("beta", 1.0)
        volatility = financials.get("volatility_252d", 0)
        institutional_ownership = financials.get("institutional_ownership", 0.5)
        
        # Use tool if: high volatility OR low institutional coverage
        # (These are conditions where news sentiment might matter more)
        if beta > 1.5 or volatility > 0.3:
            return True
        if institutional_ownership < 0.30:
            return True  # Under-discovered stocks - news might move price more
        
        return False
