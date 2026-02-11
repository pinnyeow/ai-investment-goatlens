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
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "120")),
            temperature=0.7,
        )
        self.model_name = model
    
    async def analyze(self, prompt: str, persona: str = "investment analyst", verdict: str = "hold", config: dict = None) -> str:
        """Generate analysis using LLM.
        
        Args:
            prompt: The analysis prompt
            persona: The persona to adopt
            verdict: The quantitative verdict for context
            config: Optional LangChain RunnableConfig for trace propagation.
                    When provided, the LLM span is nested under the caller's
                    trace via LangChain's parent_run_id mechanism.
        
        Returns:
            Analysis text, guaranteed to be under 150 words.
        """
        system = f"""You are {persona}. The quantitative analysis resulted in a "{verdict}" verdict.

Your job: Explain WHY this verdict makes sense based on the metrics provided. Speak naturally in first person as {persona} would.

Rules:
- No markdown, no headers, no bullet points, no asterisks
- Write 2-3 concise, direct sentences (aim for 50-100 words)
- Maximum 150 words — prioritize clarity and brevity
- Avoid filler words, unnecessary elaboration, or verbose phrasing
- Get straight to the point: explain the verdict with specific numbers
- Be punchy and direct, not flowery or wordy
- Stay consistent with the {verdict} verdict"""
        
        response = await self._llm.ainvoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            config=config,
        )
        
        # Post-process to ensure word limit (safety net)
        text = response.content or ""
        return self._truncate_to_word_limit(text, max_words=150)
    
    @staticmethod
    def _truncate_to_word_limit(text: str, max_words: int = 150) -> str:
        """
        Truncate text to a maximum word count, preserving sentence boundaries.
        
        Args:
            text: Input text
            max_words: Maximum number of words allowed
            
        Returns:
            Truncated text (guaranteed <= max_words)
        """
        if not text:
            return ""
        
        words = text.split()
        if len(words) <= max_words:
            return text
        
        # Truncate to max_words, then find the last sentence boundary
        truncated = " ".join(words[:max_words])
        
        # Try to end at a sentence boundary (period, exclamation, question mark)
        last_period = max(
            truncated.rfind("."),
            truncated.rfind("!"),
            truncated.rfind("?")
        )
        
        if last_period > max_words * 3:  # Only use if it's reasonably close
            return truncated[:last_period + 1].strip()
        
        # Otherwise, just truncate and add ellipsis
        return truncated.rstrip(".,!?") + "..."


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
