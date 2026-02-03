"""
Shared LLM Client for GOATlens agents.

Uses OpenAI GPT-4o-mini for cost-efficient analysis.
Traced automatically by Arize OpenInference instrumentation.
"""

import os
from typing import Optional
from openai import AsyncOpenAI

# Singleton instance
_client: Optional["LLMClient"] = None


class LLMClient:
    """
    Async OpenAI client wrapper for agent analysis.
    
    Usage:
        client = get_llm_client()
        response = await client.analyze(prompt, persona="Warren Buffett")
    """
    
    def __init__(self):
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "300"))
    
    async def analyze(
        self,
        prompt: str,
        persona: str = "investment analyst",
        temperature: float = 0.7,
    ) -> str:
        """
        Generate analysis using LLM.
        
        Args:
            prompt: The analysis prompt with financial data
            persona: The investor persona (e.g., "Warren Buffett")
            temperature: Creativity level (0.0-1.0)
            
        Returns:
            Generated analysis text
        """
        system_prompt = f"You are {persona}. Provide concise, insightful investment analysis in your authentic voice. Be specific about the numbers. Keep responses under 150 words."
        
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content or ""


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
