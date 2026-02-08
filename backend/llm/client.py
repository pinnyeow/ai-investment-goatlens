"""
Shared LLM Client for GOATlens agents.
Uses LangChain's ChatOpenAI for unified tracing with LangGraph.

Supports model routing: different agents can use different models based on
complexity needs. For example, Buffett (nuanced reasoning) might use gpt-4o,
while Graham (rule-based) uses gpt-4o-mini.
"""

import os
from typing import Optional, Dict
from langchain_openai import ChatOpenAI

# Cache of LLM clients by model name (for model routing)
_clients: Dict[str, "LLMClient"] = {}


class LLMClient:
    """Async LLM client wrapper using LangChain for unified tracing."""
    
    def __init__(self, model: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini"). If None, uses
                   OPENAI_MODEL env var or defaults to "gpt-4o-mini".
        """
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._llm = ChatOpenAI(
            model=model,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "200")),
            temperature=0.7,
        )
        self.model_name = model
    
    async def analyze(self, prompt: str, persona: str = "investment analyst", verdict: str = "hold") -> str:
        """Generate analysis using LLM."""
        system = f"""You are {persona}. The quantitative analysis resulted in a "{verdict}" verdict.

Your job: Explain WHY this verdict makes sense based on the metrics provided. Speak naturally in first person as {persona} would.

Rules:
- No markdown, no headers, no bullet points, no asterisks
- Write 2-3 flowing sentences
- Be specific about the numbers
- Stay consistent with the {verdict} verdict"""
        
        response = await self._llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ])
        return response.content or ""


def get_llm_client(model: Optional[str] = None) -> LLMClient:
    """
    Get or create an LLM client for the specified model.
    
    Args:
        model: Model name (e.g., "gpt-4o", "gpt-4o-mini"). If None, uses
               OPENAI_MODEL env var or defaults to "gpt-4o-mini".
    
    Returns:
        LLMClient instance (cached per model for efficiency).
    
    Example:
        # Default model (gpt-4o-mini)
        client = get_llm_client()
        
        # Specific model for nuanced reasoning
        client = get_llm_client("gpt-4o")
    """
    global _clients
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    if model not in _clients:
        _clients[model] = LLMClient(model=model)
    
    return _clients[model]
