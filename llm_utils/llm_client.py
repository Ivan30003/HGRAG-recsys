"""
LLM Client Module
Handles interactions with language model APIs (OpenAI, etc.)
Supports different LLM types for different tasks.
"""

import os
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import openai


@dataclass
class LLMResponse:
    """Structured response from LLM."""
    text: str
    tokens_used: int
    model: str
    latency: float
    cost_estimate: float


class LLMClient:
    """
    Client for interacting with various LLM APIs.
    Supports different models for different complexity tasks.
    """
    
    # Approximate costs per 1K tokens (as of 2024)
    COST_PER_1K = {
        'gpt-3.5-turbo-16k': {'input': 0.003, 'output': 0.004},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'text-davinci-003': {'input': 0.02, 'output': 0.02},
        'text-embedding-ada-002': {'input': 0.0001, 'output': 0.0},
    }
    
    def __init__(self, api_key: Optional[str] = None, default_model: str = 'gpt-3.5-turbo-16k'):
        """
        Initialize LLM client.
        
        Args:
            api_key: OpenAI API key (defaults to env variable)
            default_model: Default model to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.default_model = default_model
        
        if self.api_key:
            openai.api_key = self.api_key
        
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.call_count = 0
    
    def generate(self, 
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 model: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 2000,
                 json_mode: bool = False) -> LLMResponse:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Main prompt text
            system_prompt: Optional system-level instruction
            model: Model to use (defaults to self.default_model)
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum tokens in response
            json_mode: Whether to request JSON output
        
        Returns:
            LLMResponse with generated text and metadata
        """
        model = model or self.default_model
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        start_time = time.time()
        
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            response = openai.ChatCompletion.create(**kwargs)
            
            latency = time.time() - start_time
            output_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            # Calculate cost
            cost = self._estimate_cost(model, 
                                       response.usage.prompt_tokens,
                                       response.usage.completion_tokens)
            
            self.total_tokens_used += tokens_used
            self.total_cost += cost
            self.call_count += 1
            
            return LLMResponse(
                text=output_text,
                tokens_used=tokens_used,
                model=model,
                latency=latency,
                cost_estimate=cost
            )
            
        except Exception as e:
            print(f"LLM API error: {e}")
            # Return a fallback response
            return LLMResponse(
                text=f"Error: {str(e)}",
                tokens_used=0,
                model=model,
                latency=time.time() - start_time,
                cost_estimate=0.0
            )
    
    def generate_batch(self, 
                       prompts: List[str],
                       system_prompt: Optional[str] = None,
                       model: Optional[str] = None,
                       temperature: float = 0.3,
                       max_tokens: int = 500) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts.
        Uses parallel calls when possible.
        
        Args:
            prompts: List of prompt strings
            system_prompt: Optional system-level instruction
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
        
        Returns:
            List of LLMResponse objects
        """
        # For simplicity, process sequentially
        # In production, use async calls or batch API
        responses = []
        for prompt in prompts:
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response)
        
        return responses
    
    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost of an API call."""
        costs = self.COST_PER_1K.get(model, {'input': 0.01, 'output': 0.01})
        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']
        return input_cost + output_cost
    
    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        return {
            'total_tokens': self.total_tokens_used,
            'total_cost': self.total_cost,
            'call_count': self.call_count,
            'avg_tokens_per_call': self.total_tokens_used / max(1, self.call_count),
            'avg_cost_per_call': self.total_cost / max(1, self.call_count)
        }
    
    def reset_statistics(self):
        """Reset usage counters."""
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.call_count = 0


# Singleton instance for the application
_default_client = None

def get_llm_client(api_key: Optional[str] = None) -> LLMClient:
    """Get or create the default LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient(api_key=api_key)
    return _default_client
