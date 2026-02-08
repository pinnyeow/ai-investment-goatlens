"""
Simple memory storage for GOATlens.

Uses JSON file for persistence. In production, you'd use SQLite or a database.
This is a learning exercise for Step 9: Memory in AI Product Sense.
"""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class AnalysisRecord:
    """Record of a single analysis."""
    ticker: str
    timestamp: str
    consensus_verdict: str
    consensus_score: float
    selected_agents: List[str]
    # Store minimal data to keep memory lightweight
    key_metrics: Dict[str, Any]


class MemoryStore:
    """
    Simple memory store for user preferences and analysis history.
    
    Stores:
    - User's favorite agents (which ones they select most)
    - Last 5 analyses for comparison
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize memory store.
        
        Args:
            storage_path: Path to JSON storage file (default: backend/memory/data.json)
        """
        if storage_path is None:
            # Default to backend/memory/data.json
            backend_dir = Path(__file__).parent.parent
            storage_path = backend_dir / "memory" / "data.json"
        
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        if not self.storage_path.exists():
            return {
                "agent_preferences": {},  # {agent_name: count}
                "analyses": [],  # List of AnalysisRecord dicts
            }
        
        try:
            with open(self.storage_path, "r") as f:
                return json.load(f)
        except Exception:
            # If file is corrupted, start fresh
            return {
                "agent_preferences": {},
                "analyses": [],
            }
    
    def _save(self):
        """Save data to JSON file."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            # Log but don't fail - memory is optional
            print(f"[Memory] Failed to save: {e}")
    
    def record_analysis(
        self,
        ticker: str,
        consensus_verdict: str,
        consensus_score: float,
        selected_agents: List[str],
        key_metrics: Dict[str, Any],
    ):
        """
        Record a new analysis.
        
        Args:
            ticker: Stock ticker
            consensus_verdict: Overall verdict (e.g., "buy", "hold")
            consensus_score: Overall score
            selected_agents: List of agent names used
            key_metrics: Key financial metrics (for comparison)
        """
        record = AnalysisRecord(
            ticker=ticker,
            timestamp=datetime.now().isoformat(),
            consensus_verdict=consensus_verdict,
            consensus_score=consensus_score,
            selected_agents=selected_agents,
            key_metrics=key_metrics,
        )
        
        # Add to analyses list
        analyses = self._data.get("analyses", [])
        analyses.append(asdict(record))
        
        # Keep only last 5 analyses
        if len(analyses) > 5:
            analyses = analyses[-5:]
        
        self._data["analyses"] = analyses
        
        # Update agent preferences
        agent_prefs = self._data.get("agent_preferences", {})
        for agent in selected_agents:
            agent_prefs[agent] = agent_prefs.get(agent, 0) + 1
        
        self._data["agent_preferences"] = agent_prefs
        
        # Save to disk
        self._save()
    
    def get_favorite_agents(self, top_n: int = 3) -> List[str]:
        """
        Get user's favorite agents (most frequently selected).
        
        Args:
            top_n: Number of top agents to return
            
        Returns:
            List of agent names, sorted by frequency
        """
        agent_prefs = self._data.get("agent_preferences", {})
        if not agent_prefs:
            return []
        
        # Sort by count (descending), return top N
        sorted_agents = sorted(agent_prefs.items(), key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in sorted_agents[:top_n]]
    
    def get_recent_analyses(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent analyses for comparison.
        
        Args:
            limit: Maximum number of analyses to return
            
        Returns:
            List of analysis records (most recent first)
        """
        analyses = self._data.get("analyses", [])
        # Return most recent first
        return list(reversed(analyses[-limit:]))
    
    def get_analysis_by_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get most recent analysis for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Analysis record if found, None otherwise
        """
        analyses = self._data.get("analyses", [])
        # Search from most recent to oldest
        for analysis in reversed(analyses):
            if analysis.get("ticker", "").upper() == ticker.upper():
                return analysis
        return None
