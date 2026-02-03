"""
Shared LLM Client for GOATlens agents.
Uses LangChain's ChatOpenAI for unified tracing with LangGraph.
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI

_client: Optional["LLMClient"] = None


class LLMClient:
    """Async LLM client wrapper using LangChain for unified tracing."""
    
    def __init__(self):
        self._llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "200")),
            temperature=0.7,
        )
    
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


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
