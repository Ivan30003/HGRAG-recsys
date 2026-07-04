"""
LLM Interface Module for H-GRAGrecsys

This module provides a unified interface for interacting with various Large Language Models,
supporting multiple providers, batching, token counting, and embedding generation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Import LLM libraries (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.logger import Logger
from utils.config_loader import ConfigLoader


@dataclass
class LLMResponse:
    """
    Response from an LLM call.
    
    Attributes:
        content: Generated text content
        tokens_used: Number of tokens used
        model: Model name used
        timestamp: Response timestamp
        metadata: Additional metadata
        cost: Estimated cost of the call
        latency: Response latency in seconds
    """
    content: str
    tokens_used: int = 0
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    latency: float = 0.0


@dataclass
class LLMEmbedding:
    """
    Embedding from an LLM.
    
    Attributes:
        embedding: Embedding vector
        tokens_used: Number of tokens used
        model: Model name used
        timestamp: Response timestamp
        metadata: Additional metadata
    """
    embedding: torch.Tensor
    tokens_used: int = 0
    model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMInterface:
    """
    Unified interface for LLM interactions.
    
    This class handles:
    - Text generation with multiple LLM providers
    - Embedding generation
    - Token counting
    - Batch processing
    - Cost tracking
    - Caching
    - Rate limiting
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = Logger.get_instance(log_dir='logs', name='llm_interface')
        
        # Extract LLM configuration
        llm_config = config.get('model', {}).get('llm', {})
        self.model_name = llm_config.get('model_name', 'gpt-3.5-turbo')
        self.model_type = llm_config.get('model_type', 'openai')
        self.api_key = llm_config.get('api_key', os.getenv('OPENAI_API_KEY', ''))
        self.api_base = llm_config.get('api_base', '')
        self.max_tokens = llm_config.get('max_tokens', 1000)
        self.temperature = llm_config.get('temperature', 0.7)
        self.top_p = llm_config.get('top_p', 0.9)
        self.frequency_penalty = llm_config.get('frequency_penalty', 0.0)
        self.presence_penalty = llm_config.get('presence_penalty', 0.0)
        self.timeout = llm_config.get('timeout', 30)
        self.retry_count = llm_config.get('retry_count', 3)
        self.retry_delay = llm_config.get('retry_delay', 1.0)
        
        # Caching configuration
        self.use_cache = llm_config.get('use_cache', True)
        self.cache_size = llm_config.get('cache_size', 1000)
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting
        self.rate_limit = llm_config.get('rate_limit', 100)
        self.rate_limit_period = llm_config.get('rate_limit_period', 60)
        self.request_timestamps: List[float] = []
        
        # Cost tracking
        self.cost_tracker = {
            'total_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'calls_by_model': defaultdict(int),
            'tokens_by_model': defaultdict(int),
            'cost_by_model': defaultdict(float)
        }
        
        # Initialize system prompt
        self.system_prompt = llm_config.get('system_prompt', '')
        
        # Initialize provider
        self._initialize_provider()
        
        # Embedding model
        self.embedding_model = llm_config.get('embedding_model', 'text-embedding-ada-002')
        self.embedding_dim = llm_config.get('embedding_dim', 1536)
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        self.logger.log_info(f"Initialized LLMInterface with model={self.model_name}, type={self.model_type}")
    
    def _initialize_provider(self) -> None:
        """
        Initialize the LLM provider based on configuration.
        """
        if self.model_type == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            
            openai.api_key = self.api_key
            if self.api_base:
                openai.api_base = self.api_base
            
            self.logger.log_info("Initialized OpenAI provider")
        
        elif self.model_type == 'anthropic':
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not installed. Install with: pip install anthropic")
            
            self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
            self.logger.log_info("Initialized Anthropic provider")
        
        elif self.model_type == 'huggingface':
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library not installed. Install with: pip install transformers")
            
            self._initialize_huggingface()
            self.logger.log_info("Initialized HuggingFace provider")
        
        elif self.model_type == 'local':
            self._initialize_local()
            self.logger.log_info("Initialized local provider")
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _initialize_huggingface(self) -> None:
        """
        Initialize HuggingFace models.
        """
        self.logger.log_info(f"Loading HuggingFace model: {self.model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load embedding model if different
        if self.embedding_model and self.embedding_model != self.model_name:
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model)
            self.embedding_model_obj = AutoModel.from_pretrained(
                self.embedding_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.embedding_tokenizer = self.tokenizer
            self.embedding_model_obj = self.model
    
    def _initialize_local(self) -> None:
        """
        Initialize local model (placeholder for custom local models).
        """
        self.logger.log_info(f"Loading local model: {self.model_name}")
        # Implement local model loading here if needed
        # This is a placeholder for custom local LLM implementations
    
    def generate(self, prompt: str, 
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                stop: Optional[Union[str, List[str]]] = None,
                **kwargs) -> LLMResponse:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse: Generated response
        """
        self.logger.log_info(f"Generating text from prompt (length: {len(prompt)})")
        
        # Check cache
        cache_key = self._get_cache_key(prompt, max_tokens, temperature, top_p, stop)
        if self.use_cache and cache_key in self.cache:
            self.logger.log_info("Cache hit for generation")
            cached_response = self.cache[cache_key]
            return LLMResponse(
                content=cached_response['content'],
                tokens_used=cached_response['tokens_used'],
                model=self.model_name,
                metadata=cached_response.get('metadata', {}),
                cost=cached_response.get('cost', 0.0),
                latency=cached_response.get('latency', 0.0)
            )
        
        # Apply rate limiting
        self._apply_rate_limit()
        
        # Set parameters
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p or self.top_p
        
        # Generate
        start_time = time.time()
        response = None
        for attempt in range(self.retry_count):
            try:
                response = self._generate_impl(
                    prompt, max_tokens, temperature, top_p, stop, **kwargs
                )
                break
            except Exception as e:
                self.logger.log_warning(f"Generation attempt {attempt+1} failed: {str(e)}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
        
        if response is None:
            raise RuntimeError("Failed to generate response after retries")
        
        # Calculate cost
        cost = self._calculate_cost(response['tokens_used'])
        
        # Create response object
        llm_response = LLMResponse(
            content=response['content'],
            tokens_used=response['tokens_used'],
            model=self.model_name,
            metadata=response.get('metadata', {}),
            cost=cost,
            latency=time.time() - start_time
        )
        
        # Track costs
        self._track_cost(llm_response)
        
        # Cache response
        if self.use_cache:
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = {
                'content': response['content'],
                'tokens_used': response['tokens_used'],
                'metadata': response.get('metadata', {}),
                'cost': cost,
                'latency': llm_response.latency
            }
        
        self.logger.log_info(f"Generated {response['tokens_used']} tokens in {llm_response.latency:.2f}s")
        return llm_response
    
    def _generate_impl(self, prompt: str, max_tokens: int, 
                       temperature: float, top_p: float,
                       stop: Optional[Union[str, List[str]]],
                       **kwargs) -> Dict[str, Any]:
        """
        Provider-specific generation implementation.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Generated content and metadata
        """
        if self.model_type == 'openai':
            return self._generate_openai(prompt, max_tokens, temperature, top_p, stop, **kwargs)
        elif self.model_type == 'anthropic':
            return self._generate_anthropic(prompt, max_tokens, temperature, top_p, stop, **kwargs)
        elif self.model_type == 'huggingface':
            return self._generate_huggingface(prompt, max_tokens, temperature, top_p, stop, **kwargs)
        elif self.model_type == 'local':
            return self._generate_local(prompt, max_tokens, temperature, top_p, stop, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _generate_openai(self, prompt: str, max_tokens: int,
                         temperature: float, top_p: float,
                         stop: Optional[Union[str, List[str]]],
                         **kwargs) -> Dict[str, Any]:
        """
        Generate using OpenAI API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional OpenAI parameters
            
        Returns:
            Dict[str, Any]: Generated content and metadata
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare parameters
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": self.timeout
        }
        
        if stop:
            params["stop"] = stop
        
        # Add any extra parameters
        params.update(kwargs)
        
        # Make API call
        try:
            response = openai.ChatCompletion.create(**params)
            
            return {
                'content': response.choices[0].message.content,
                'tokens_used': response.usage.total_tokens,
                'metadata': {
                    'model': response.model,
                    'finish_reason': response.choices[0].finish_reason,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens
                }
            }
        except Exception as e:
            self.logger.log_error(f"OpenAI API error: {str(e)}")
            raise
    
    def _generate_anthropic(self, prompt: str, max_tokens: int,
                           temperature: float, top_p: float,
                           stop: Optional[Union[str, List[str]]],
                           **kwargs) -> Dict[str, Any]:
        """
        Generate using Anthropic API.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional Anthropic parameters
            
        Returns:
            Dict[str, Any]: Generated content and metadata
        """
        # Prepare system prompt
        system = self.system_prompt if self.system_prompt else None
        
        # Prepare parameters
        params = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        
        if system:
            params["system"] = system
        
        if stop:
            params["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        
        params.update(kwargs)
        
        try:
            response = self.anthropic_client.messages.create(
                **params,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                'content': response.content[0].text,
                'tokens_used': response.usage.output_tokens + response.usage.input_tokens,
                'metadata': {
                    'model': response.model,
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            }
        except Exception as e:
            self.logger.log_error(f"Anthropic API error: {str(e)}")
            raise
    
    def _generate_huggingface(self, prompt: str, max_tokens: int,
                             temperature: float, top_p: float,
                             stop: Optional[Union[str, List[str]]],
                             **kwargs) -> Dict[str, Any]:
        """
        Generate using HuggingFace models.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional HuggingFace parameters
            
        Returns:
            Dict[str, Any]: Generated content and metadata
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_length = inputs['input_ids'].shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove original prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        # Count tokens
        tokens_used = len(outputs[0]) - input_length
        
        return {
            'content': generated_text,
            'tokens_used': tokens_used,
            'metadata': {
                'model': self.model_name,
                'input_tokens': input_length,
                'output_tokens': tokens_used
            }
        }
    
    def _generate_local(self, prompt: str, max_tokens: int,
                       temperature: float, top_p: float,
                       stop: Optional[Union[str, List[str]]],
                       **kwargs) -> Dict[str, Any]:
        """
        Generate using local model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Generated content and metadata
        """
        # Placeholder for custom local model implementation
        # This should be overridden with actual local model inference
        raise NotImplementedError("Local model generation not implemented")
    
    def get_embedding(self, text: Union[str, List[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Get embeddings for text(s).
        
        Args:
            text: Single text or list of texts
            
        Returns:
            Union[torch.Tensor, List[torch.Tensor]]: Embedding vector(s)
        """
        if isinstance(text, str):
            return self._get_single_embedding(text)
        else:
            return self._get_batch_embeddings(text)
    
    def _get_single_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            torch.Tensor: Embedding vector
        """
        if self.model_type == 'openai':
            return self._get_embedding_openai(text)
        elif self.model_type in ['huggingface', 'local']:
            return self._get_embedding_huggingface(text)
        else:
            raise ValueError(f"Unsupported model type for embeddings: {self.model_type}")
    
    def _get_embedding_openai(self, text: str) -> torch.Tensor:
        """
        Get embedding using OpenAI API.
        
        Args:
            text: Input text
            
        Returns:
            torch.Tensor: Embedding vector
        """
        try:
            response = openai.Embedding.create(
                model=self.embedding_model,
                input=text
            )
            
            embedding = np.array(response.data[0].embedding)
            return torch.tensor(embedding, dtype=torch.float32)
        except Exception as e:
            self.logger.log_error(f"OpenAI embedding error: {str(e)}")
            raise
    
    def _get_embedding_huggingface(self, text: str) -> torch.Tensor:
        """
        Get embedding using HuggingFace model.
        
        Args:
            text: Input text
            
        Returns:
            torch.Tensor: Embedding vector
        """
        # Tokenize
        inputs = self.embedding_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.embedding_model_obj(**inputs)
            # Use mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        return embedding
    
    def _get_batch_embeddings(self, texts: List[str]) -> List[torch.Tensor]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List[torch.Tensor]: List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self._get_single_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def batch_generate(self, prompts: List[str], 
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None,
                      top_p: Optional[float] = None,
                      **kwargs) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional parameters
            
        Returns:
            List[LLMResponse]: List of generated responses
        """
        self.logger.log_info(f"Batch generating for {len(prompts)} prompts")
        
        responses = []
        for i, prompt in enumerate(prompts):
            self.logger.log_info(f"Processing prompt {i+1}/{len(prompts)}")
            response = self.generate(prompt, max_tokens, temperature, top_p, **kwargs)
            responses.append(response)
        
        return responses
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
            
        Returns:
            int: Number of tokens
        """
        if self.model_type == 'openai':
            # Use tiktoken for OpenAI
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model(self.model_name)
                return len(encoding.encode(text))
            except:
                # Fallback to approximate count
                return len(text.split()) * 1.3
        
        elif self.model_type in ['huggingface', 'local']:
            if hasattr(self, 'tokenizer'):
                return len(self.tokenizer.encode(text))
        
        # Fallback: approximate token count
        return len(text.split()) * 1.3
    
    def set_system_prompt(self, system_prompt: str) -> None:
        """
        Set the system prompt for generation.
        
        Args:
            system_prompt: New system prompt
        """
        self.system_prompt = system_prompt
        self.logger.log_info(f"System prompt updated (length: {len(system_prompt)})")
    
    def _apply_rate_limit(self) -> None:
        """
        Apply rate limiting to API calls.
        """
        current_time = time.time()
        # Remove old timestamps
        self.request_timestamps = [
            t for t in self.request_timestamps 
            if current_time - t < self.rate_limit_period
        ]
        
        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = self.rate_limit_period - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                self.logger.log_warning(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        self.request_timestamps.append(current_time)
    
    def _get_cache_key(self, prompt: str, max_tokens: Optional[int],
                      temperature: Optional[float], top_p: Optional[float],
                      stop: Optional[Union[str, List[str]]]) -> str:
        """
        Generate cache key for a request.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens
            temperature: Temperature
            top_p: Top-p value
            stop: Stop sequences
            
        Returns:
            str: Cache key
        """
        key_data = {
            'prompt': prompt,
            'model': self.model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': str(stop) if stop else None
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _calculate_cost(self, tokens: int) -> float:
        """
        Calculate cost for token usage.
        
        Args:
            tokens: Number of tokens
            
        Returns:
            float: Estimated cost
        """
        # Cost per 1000 tokens for different models
        cost_rates = {
            'gpt-3.5-turbo': 0.002,
            'gpt-3.5-turbo-16k': 0.003,
            'gpt-4': 0.06,
            'gpt-4-32k': 0.12,
            'claude-2': 0.008,
            'claude-3': 0.015
        }
        
        rate = cost_rates.get(self.model_name, 0.01)
        return (tokens / 1000) * rate
    
    def _track_cost(self, response: LLMResponse) -> None:
        """
        Track cost statistics.
        
        Args:
            response: LLMResponse object
        """
        self.cost_tracker['total_calls'] += 1
        self.cost_tracker['total_tokens'] += response.tokens_used
        self.cost_tracker['total_cost'] += response.cost
        self.cost_tracker['calls_by_model'][response.model] += 1
        self.cost_tracker['tokens_by_model'][response.model] += response.tokens_used
        self.cost_tracker['cost_by_model'][response.model] += response.cost
    
    def get_cost_statistics(self) -> Dict[str, Any]:
        """
        Get cost statistics.
        
        Returns:
            Dict[str, Any]: Cost statistics
        """
        stats = self.cost_tracker.copy()
        stats['average_cost_per_call'] = (
            stats['total_cost'] / stats['total_calls'] 
            if stats['total_calls'] > 0 else 0.0
        )
        stats['average_tokens_per_call'] = (
            stats['total_tokens'] / stats['total_calls'] 
            if stats['total_calls'] > 0 else 0.0
        )
        return stats
    
    def clear_cache(self) -> None:
        """Clear the generation cache."""
        self.cache.clear()
        self.logger.log_info("Cleared generation cache")
    
    def reset_cost_tracking(self) -> None:
        """Reset cost tracking statistics."""
        self.cost_tracker = {
            'total_calls': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'calls_by_model': defaultdict(int),
            'tokens_by_model': defaultdict(int),
            'cost_by_model': defaultdict(float)
        }
        self.logger.log_info("Reset cost tracking")
    
    def estimate_cost(self, text_length: int, max_tokens: int) -> float:
        """
        Estimate cost for a generation request.
        
        Args:
            text_length: Length of input text in characters
            max_tokens: Maximum tokens to generate
            
        Returns:
            float: Estimated cost
        """
        # Approximate input tokens (rough estimate)
        input_tokens = text_length / 4  # Rough estimate
        total_tokens = input_tokens + max_tokens
        return self._calculate_cost(total_tokens)
    
    async def async_generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Async version of generate.
        
        Args:
            prompt: Input prompt
            **kwargs: Generation parameters
            
        Returns:
            LLMResponse: Generated response
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate,
            prompt,
            **kwargs
        )
    
    async def async_batch_generate(self, prompts: List[str], **kwargs) -> List[LLMResponse]:
        """
        Async batch generation.
        
        Args:
            prompts: List of prompts
            **kwargs: Generation parameters
            
        Returns:
            List[LLMResponse]: List of responses
        """
        tasks = [self.async_generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def __str__(self) -> str:
        """String representation."""
        return (f"LLMInterface(model={self.model_name}, type={self.model_type}, "
                f"calls={self.cost_tracker['total_calls']}, "
                f"cost=${self.cost_tracker['total_cost']:.4f})")


class LLMFactory:
    """
    Factory for creating LLM interfaces.
    
    This class provides a factory pattern for creating LLM interfaces
    with different configurations.
    """
    
    @staticmethod
    def create_llm(model_type: str, config: Dict[str, Any]) -> LLMInterface:
        """
        Create an LLM interface.
        
        Args:
            model_type: Type of LLM ('openai', 'anthropic', 'huggingface', 'local')
            config: Configuration dictionary
            
        Returns:
            LLMInterface: LLM interface instance
        """
        # Update config with model type
        if 'model' not in config:
            config['model'] = {}
        if 'llm' not in config['model']:
            config['model']['llm'] = {}
        
        config['model']['llm']['model_type'] = model_type
        
        return LLMInterface(config)
    
    @staticmethod
    def create_openai_llm(model_name: str = 'gpt-3.5-turbo', 
                         api_key: Optional[str] = None,
                         **kwargs) -> LLMInterface:
        """
        Create an OpenAI LLM interface.
        
        Args:
            model_name: OpenAI model name
            api_key: OpenAI API key
            **kwargs: Additional configuration
            
        Returns:
            LLMInterface: OpenAI LLM interface
        """
        config = {
            'model': {
                'llm': {
                    'model_type': 'openai',
                    'model_name': model_name,
                    'api_key': api_key or os.getenv('OPENAI_API_KEY', ''),
                    **kwargs
                }
            }
        }
        return LLMInterface(config)
    
    @staticmethod
    def create_huggingface_llm(model_name: str, 
                              embedding_model: Optional[str] = None,
                              **kwargs) -> LLMInterface:
        """
        Create a HuggingFace LLM interface.
        
        Args:
            model_name: HuggingFace model name
            embedding_model: HuggingFace embedding model name
            **kwargs: Additional configuration
            
        Returns:
            LLMInterface: HuggingFace LLM interface
        """
        config = {
            'model': {
                'llm': {
                    'model_type': 'huggingface',
                    'model_name': model_name,
                    'embedding_model': embedding_model or model_name,
                    **kwargs
                }
            }
        }
        return LLMInterface(config)


# Example usage
if __name__ == "__main__":
    # Load configuration
    config_path = "config/default_config.yaml"
    config_loader = ConfigLoader(config_path)
    config = config_loader.load_config()
    
    # Create LLM interface
    llm = LLMInterface(config)
    
    # Test generation (if API key is available)
    if OPENAI_AVAILABLE and llm.api_key:
        print("Testing generation...")
        response = llm.generate(
            "What is the capital of France?",
            max_tokens=50,
            temperature=0.5
        )
        print(f"Response: {response.content}")
        print(f"Tokens: {response.tokens_used}")
        print(f"Cost: ${response.cost:.6f}")
        
        # Test embedding
        embedding = llm.get_embedding("Hello world")
        print(f"Embedding shape: {embedding.shape}")
        
        # Test batch generation
        prompts = [
            "What is the capital of France?",
            "What is the largest planet?",
            "Who wrote Romeo and Juliet?"
        ]
        responses = llm.batch_generate(prompts, max_tokens=30)
        for i, response in enumerate(responses):
            print(f"Response {i+1}: {response.content[:50]}...")
        
        # Get cost statistics
        stats = llm.get_cost_statistics()
        print(f"Cost statistics: {stats}")
    
    # Create using factory
    if OPENAI_AVAILABLE:
        llm_factory = LLMFactory.create_openai_llm(
            model_name='gpt-3.5-turbo',
            temperature=0.8
        )
        print(f"Created OpenAI LLM: {llm_factory}")