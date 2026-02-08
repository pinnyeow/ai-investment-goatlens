"""
Memory module for GOATlens.

Stores user preferences and analysis history for personalized experience.
This demonstrates Step 9: Memory in AI Product Sense.

Features:
- User agent preferences (which agents they select most)
- Last 5 analyses for comparison
- Simple JSON-based storage (can be upgraded to SQLite later)
"""

from .storage import MemoryStore, AnalysisRecord

__all__ = ["MemoryStore", "AnalysisRecord"]
