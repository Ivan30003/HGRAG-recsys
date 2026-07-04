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

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    TextStreamer,
    StoppingCriteria,
    StoppingCriteriaList
)
from transformers.cache_utils import DynamicCache



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
        'gpt-5.2': {'input': 0.005, 'output': 0.01},
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


def get_llm_client(config: dict) -> LLMClient:
    """Get or create the default LLM client."""
    if config['llm'].get('path'):
        print(f"LLM")
        return Qwen3LocalLLM(config['llm'].get('path'))
    else:
        return LLMClient(config['llm'].get('api_key'))
    



# *************************************************************




"""
Local LLM Client Module for Qwen3 Model Family
Provides optimized local inference for agent memory generation and reflection.
Supports Qwen3 models with efficient memory management and structured generation.
"""


logger = logging.getLogger(__name__)


class Qwen3ModelVariant(Enum):
    """Available Qwen3 model variants."""
    QWEN3_0_6B = "Qwen/Qwen3-0.6B"
    QWEN3_1_7B = "Qwen/Qwen3-1.7B"
    QWEN3_4B = "Qwen/Qwen3-4B"
    QWEN3_8B = "Qwen/Qwen3-8B"
    QWEN3_14B = "Qwen/Qwen3-14B"
    QWEN3_32B = "Qwen/Qwen3-32B"
    QWEN3_72B = "Qwen/Qwen3-72B"
    # Instruct variants
    QWEN3_1_7B_INSTRUCT = "Qwen/Qwen3-1.7B-Instruct"
    QWEN3_4B_INSTRUCT = "Qwen/Qwen3-4B-Instruct"
    QWEN3_8B_INSTRUCT = "Qwen/Qwen3-8B-Instruct"
    QWEN3_14B_INSTRUCT = "Qwen/Qwen3-14B-Instruct"
    QWEN3_32B_INSTRUCT = "Qwen/Qwen3-32B-Instruct"
    QWEN3_72B_INSTRUCT = "Qwen/Qwen3-72B-Instruct"


class AgentMemoryStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for agent memory generation."""
    
    def __init__(self, stop_tokens: List[int], max_length: int):
        super().__init__()
        self.stop_tokens = stop_tokens
        self.max_length = max_length
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[-1] >= self.max_length:
            return True
        for stop_token in self.stop_tokens:
            if input_ids[0, -1].item() == stop_token:
                return True
        return False


@dataclass
class LocalLLMResponse:
    """Structured response from local LLM."""
    text: str
    tokens_generated: int
    total_tokens: int
    generation_time: float
    tokens_per_second: float
    model_name: str
    finish_reason: str  # 'stop', 'length', 'stop_token'
    memory_usage_mb: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'tokens_generated': self.tokens_generated,
            'total_tokens': self.total_tokens,
            'generation_time': self.generation_time,
            'tokens_per_second': self.tokens_per_second,
            'model_name': self.model_name,
            'finish_reason': self.finish_reason,
            'memory_usage_mb': self.memory_usage_mb
        }


@dataclass
class Qwen3GenerationConfig:
    """Configuration for Qwen3 text generation optimized for agent memory."""
    
    # Basic generation parameters
    temperature: float = 0.3
    top_p: float = 0.85
    top_k: int = 50
    max_new_tokens: int = 512
    min_new_tokens: int = 10
    repetition_penalty: float = 1.05
    length_penalty: float = 1.0
    
    # Qwen3-specific parameters
    use_cache: bool = True
    do_sample: bool = True
    early_stopping: bool = False
    num_beams: int = 1  # 1 for sampling, >1 for beam search
    
    # Memory-specific parameters
    presence_penalty: float = 0.1  # Encourage diversity in memories
    frequency_penalty: float = 0.1  # Reduce repetition
    no_repeat_ngram_size: int = 3  # Prevent repetitive phrases
    
    # Structured output
    output_format: str = "text"  # 'text', 'json', 'markdown'
    
    # Performance
    use_flash_attention: bool = False
    use_vllm: bool = False  # Use vLLM for faster inference
    prefill_chunk_size: int = 4096


class Qwen3LocalLLM:
    """
    Local LLM client optimized for Qwen3 model family.
    
    Designed specifically for Hybrid-GraphRAG agent memory operations:
    - Memory initialization and summarization
    - Collaborative reflection
    - Preference extraction
    - Neighborhood propagation
    - Memory fusion
    
    Features:
    - 4-bit and 8-bit quantization support
    - Flash Attention 2 for faster inference
    - Structured JSON output
    - Memory-efficient KV caching
    - Batched inference for multiple agents
    - Streaming generation for long reflections
    """
    
    # Default system prompts for different memory tasks
    SYSTEM_PROMPTS = {
        'memory_init': """You are an AI agent creating a concise, personalized memory profile.
Focus on distinguishing characteristics and specific preferences.
Be precise and avoid generic statements.
Use clear, structured language.""",
        
        'reflection': """You are an AI agent reflecting on an interaction to improve your understanding.
Analyze what went wrong or right in the decision.
Identify specific patterns to learn or adjust.
Be honest about mistakes and clear about corrections.
Focus on actionable insights for memory update.""",
        
        'fusion': """You are an AI agent fusing multiple perspectives into coherent understanding.
Synthesize information from various sources.
Resolve contradictions logically.
Maintain consistency with core identity.
Be concise and specific in the fused output.""",
        
        'propagation': """You are an AI agent receiving signals from similar agents.
Incorporate relevant new patterns from neighbors.
Maintain your unique characteristics.
Only adopt patterns consistent with your identity.
Be selective in what you learn from others.""",
        
        'decision': """You are an AI agent making a recommendation decision.
Base your decision on your memory and preferences.
Consider both positive and negative patterns.
Explain your reasoning clearly.
Be confident but acknowledge uncertainty.""",
        
        'ranking': """You are an AI agent ranking items for recommendation.
Consider user preferences, collaborative signals, and item features.
Provide clear, personalized explanations for each ranking.
Focus on the most relevant items for the user."""
    }
    
    def __init__(self,
                 model_variant: Union[str, Qwen3ModelVariant] = Qwen3ModelVariant.QWEN3_4B_INSTRUCT,
                 device: str = "auto",
                 quantization: str = "none",  # '4bit', '8bit', 'none'
                 generation_config: Optional[Qwen3GenerationConfig] = None):
        """
        Initialize Qwen3 local LLM client.
        
        Args:
            model_variant: Qwen3 model variant to use
            device: Device for inference ('auto', 'cuda', 'cpu', 'mps')
            quantization: Quantization method ('4bit', '8bit', 'none')
            max_memory: GPU memory limits per device
            generation_config: Generation configuration
            cache_dir: Directory for model cache
            offload_folder: Directory for offloaded weights
        """
        self.model_variant = model_variant.value if isinstance(model_variant, Qwen3ModelVariant) else model_variant
        self.quantization = quantization
        self.generation_config = generation_config or Qwen3GenerationConfig()
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing Qwen3 model: {self.model_variant}")
        logger.info(f"Device: {self.device}, Quantization: {quantization}")
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_tokens_generated': 0,
            'total_time': 0.0,
            'avg_tokens_per_second': 0.0,
            'peak_memory_mb': 0.0,
            'errors': 0
        }

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
        # Set up generation parameters
        self._setup_generation_params()
        
        # Special tokens
        self.special_tokens = {
            'system_start': '<|im_start|>system\n',
            'system_end': '<|im_end|>\n',
            'user_start': '<|im_start|>user\n',
            'user_end': '<|im_end|>\n',
            'assistant_start': '<|im_start|>assistant\n',
            'assistant_end': '<|im_end|>\n',
            'memory_start': '<|memory_start|>',
            'memory_end': '<|memory_end|>',
            'json_start': '<|json_start|>',
            'json_end': '<|json_end|>'
        }
        
        # Conversation history cache
        self._conversation_cache: Dict[str, List[Dict]] = {}
        
        logger.info(f"Qwen3 model loaded successfully")
    
    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load Qwen3 model and tokenizer with appropriate configuration.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_variant,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization
        model_kwargs = {
            'trust_remote_code': True,
            'device_map': self.device if self.device != 'cpu' else None,
        }
        
        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model_kwargs['quantization_config'] = bnb_config
            logger.info("Using 4-bit NF4 quantization")
            
        elif self.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model_kwargs['quantization_config'] = bnb_config
            logger.info("Using 8-bit quantization")
        
        elif self.quantization == "none":
            if self.device == "cuda":
                model_kwargs['torch_dtype'] = torch.bfloat16
            logger.info("Using full precision (no quantization)")
        
        # Try to use Flash Attention 2 if available
        if self.generation_config.use_flash_attention:
            try:
                model_kwargs['attn_implementation'] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except Exception:
                logger.warning("Flash Attention 2 not available, using default attention")
        
        # Load model
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_variant,
            **model_kwargs
        )
        
        if self.device == "cpu" and self.quantization == "none":
            model = model.to(self.device)
        
        # Record memory usage
        if torch.cuda.is_available():
            self.stats['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return model, tokenizer
    
    def _setup_generation_params(self):
        """Set up default generation parameters."""
        self.default_generation_kwargs = {
            'temperature': self.generation_config.temperature,
            'top_p': self.generation_config.top_p,
            'top_k': self.generation_config.top_k,
            'max_new_tokens': self.generation_config.max_new_tokens,
            'min_new_tokens': self.generation_config.min_new_tokens,
            'repetition_penalty': self.generation_config.repetition_penalty,
            'length_penalty': self.generation_config.length_penalty,
            'do_sample': self.generation_config.do_sample,
            'use_cache': self.generation_config.use_cache,
            'early_stopping': self.generation_config.early_stopping,
            'num_beams': self.generation_config.num_beams,
            'no_repeat_ngram_size': self.generation_config.no_repeat_ngram_size,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Presence and frequency penalties for PyTorch >= 2.0
        if hasattr(self.model, 'generation_config'):
            self.model.generation_config.presence_penalty = self.generation_config.presence_penalty
            self.model.generation_config.frequency_penalty = self.generation_config.frequency_penalty
    
    def _build_chat_prompt(self,
                           messages: List[Dict[str, str]],
                           system_prompt: Optional[str] = None,
                           output_format: str = "text") -> str:
        """
        Build Qwen3 chat format prompt.
        
        Qwen3 uses the ChatML format:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_response}<|im_end|>
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt override
            output_format: Desired output format
        
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # System message
        if system_prompt:
            prompt_parts.append(
                f"{self.special_tokens['system_start']}{system_prompt}{self.special_tokens['system_end']}"
            )
        
        # Conversation messages
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(
                    f"{self.special_tokens['system_start']}{content}{self.special_tokens['system_end']}"
                )
            elif role == 'user':
                prompt_parts.append(
                    f"{self.special_tokens['user_start']}{content}{self.special_tokens['user_end']}"
                )
            elif role == 'assistant':
                prompt_parts.append(
                    f"{self.special_tokens['assistant_start']}{content}{self.special_tokens['assistant_end']}"
                )
        
        # Add output format hint
        if output_format == "json":
            prompt_parts.append(
                f"{self.special_tokens['json_start']}\n"
            )
        elif output_format == "memory":
            prompt_parts.append(
                f"{self.special_tokens['memory_start']}\n"
            )
        
        # Start assistant response
        prompt_parts.append(f"{self.special_tokens['assistant_start']}")
        
        return "".join(prompt_parts)
    
    def generate(self,
                 prompt: str,
                 system_prompt: Optional[str] = None,
                 max_new_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 output_format: str = "text",
                 stop_sequences: Optional[List[str]] = None,
                 stream: bool = False,
                 use_chat_template: bool = True) -> LocalLLMResponse:
        """
        Generate text response from the model.
        
        Args:
            prompt: Input prompt text
            system_prompt: Optional system prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            output_format: Output format ('text', 'json', 'memory')
            stop_sequences: Custom stop sequences
            stream: Whether to stream output
            use_chat_template: Whether to use chat template
        
        Returns:
            LocalLLMResponse with generated text and metadata
        """
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Build full prompt
            if use_chat_template:
                messages = [{'role': 'user', 'content': prompt}]
                full_prompt = self._build_chat_prompt(
                    messages, system_prompt, output_format
                )
            else:
                full_prompt = prompt
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=8192 - (max_new_tokens or self.generation_config.max_new_tokens)
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            input_length = inputs['input_ids'].shape[1]
            
            # Set up generation kwargs
            gen_kwargs = self.default_generation_kwargs.copy()
            
            if max_new_tokens:
                gen_kwargs['max_new_tokens'] = max_new_tokens
            
            if temperature is not None:
                gen_kwargs['temperature'] = temperature
                gen_kwargs['do_sample'] = temperature > 0
            
            # Set up stopping criteria
            stopping_criteria = None
            if stop_sequences:
                stop_token_ids = []
                for seq in stop_sequences:
                    ids = self.tokenizer.encode(seq, add_special_tokens=False)
                    stop_token_ids.extend(ids)
                
                if stop_token_ids:
                    stopping_criteria = StoppingCriteriaList([
                        AgentMemoryStoppingCriteria(
                            stop_token_ids,
                            gen_kwargs['max_new_tokens']
                        )
                    ])
                    gen_kwargs['stopping_criteria'] = stopping_criteria
            
            # Streamer for real-time output
            streamer = None
            if stream:
                streamer = TextStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                gen_kwargs['streamer'] = streamer
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=False
                )
            
            # Decode output
            generated_ids = outputs.sequences[0][input_length:]
            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Remove stop sequences
            if stop_sequences:
                for seq in stop_sequences:
                    generated_text = generated_text.replace(seq, '')
            
            # Clean up output based on format
            generated_text = self._clean_output(generated_text, output_format)
            
            # Compute statistics
            generation_time = time.time() - start_time
            tokens_generated = len(generated_ids)
            total_tokens = input_length + tokens_generated
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            self.stats['total_tokens_generated'] += tokens_generated
            self.stats['total_time'] += generation_time
            
            # Determine finish reason
            if tokens_generated >= gen_kwargs['max_new_tokens']:
                finish_reason = 'length'
            elif stopping_criteria and hasattr(stopping_criteria, '_called'):
                finish_reason = 'stop_token'
            else:
                finish_reason = 'stop'
            
            return LocalLLMResponse(
                text=generated_text.strip(),
                tokens_generated=tokens_generated,
                total_tokens=total_tokens,
                generation_time=generation_time,
                tokens_per_second=tokens_per_second,
                model_name=self.model_variant,
                finish_reason=finish_reason,
                memory_usage_mb=self.stats['peak_memory_mb']
            )
            
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Generation error: {e}")
            return LocalLLMResponse(
                text=f"Error: {str(e)}",
                tokens_generated=0,
                total_tokens=0,
                generation_time=time.time() - start_time,
                tokens_per_second=0.0,
                model_name=self.model_variant,
                finish_reason='error',
                memory_usage_mb=self.stats['peak_memory_mb']
            )
    
    def _clean_output(self, text: str, output_format: str) -> str:
        """
        Clean generated output based on format.
        
        Args:
            text: Raw generated text
            output_format: Output format
        
        Returns:
            Cleaned text
        """
        # Remove special tokens
        for token_name, token_value in self.special_tokens.items():
            text = text.replace(token_value, '')
        
        # Remove trailing assistant markers
        text = text.replace('<|im_end|>', '').strip()
        
        # Format-specific cleaning
        if output_format == "json":
            # Try to extract JSON block
            if '```json' in text:
                start = text.find('```json') + 7
                end = text.find('```', start)
                if end > start:
                    text = text[start:end].strip()
            elif '{' in text and '}' in text:
                start = text.find('{')
                end = text.rfind('}') + 1
                text = text[start:end].strip()
        
        elif output_format == "memory":
            # Extract memory content
            if 'Memory:' in text:
                text = text.split('Memory:', 1)[1].strip()
            elif 'Updated memory:' in text:
                text = text.split('Updated memory:', 1)[1].strip()
        
        return text.strip()
    
    def generate_with_json_output(self,
                                   prompt: str,
                                   system_prompt: Optional[str] = None,
                                   max_new_tokens: int = 1024,
                                   temperature: float = 0.2,
                                   retry_on_error: bool = True,
                                   max_retries: int = 3) -> Dict:
        """
        Generate response with guaranteed JSON output.
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt
            max_new_tokens: Maximum tokens
            temperature: Temperature (lower for JSON)
            retry_on_error: Whether to retry on JSON parse error
            max_retries: Maximum retry attempts
        
        Returns:
            Parsed JSON dictionary
        """
        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + "\n\nYou MUST respond with valid JSON only. No other text."
        
        for attempt in range(max_retries):
            response = self.generate(
                prompt=prompt,
                system_prompt=json_system,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                output_format="json"
            )
            
            try:
                # Try to parse as JSON
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError:
                if attempt < max_retries - 1 and retry_on_error:
                    logger.warning(f"JSON parse failed, retrying (attempt {attempt + 1})")
                    # Add error feedback to prompt
                    prompt = (
                        f"Your previous response was not valid JSON. "
                        f"Please respond with valid JSON only.\n\nOriginal prompt:\n{prompt}"
                    )
                    temperature = min(0.5, temperature + 0.1)  # Slightly increase temperature
                else:
                    logger.error(f"Failed to parse JSON after {max_retries} attempts")
                    return {"error": "json_parse_failed", "raw_text": response.text[:500]}
        
        return {"error": "max_retries_exceeded"}
    
    def generate_memory_summary(self,
                                 agent_data: Dict,
                                 memory_type: str = "intrinsic",
                                 max_words: int = 80) -> str:
        """
        Generate a concise memory summary for an agent.
        
        Args:
            agent_data: Agent data dictionary
            memory_type: Type of memory ('intrinsic', 'collaborative', 'interaction')
            max_words: Maximum words in summary
        
        Returns:
            Memory summary text
        """
        system_prompt = self.SYSTEM_PROMPTS['memory_init']
        
        prompt = f"""Generate a concise {memory_type} memory summary for an agent.

Agent Data:
{json.dumps(agent_data, indent=2)}

Requirements:
1. Keep the summary under {max_words} words
2. Focus on distinguishing, specific characteristics
3. Avoid generic statements
4. Include both preferences and constraints
5. Use clear, structured language

Memory Type: {memory_type}

Generate the memory summary:"""
        
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=256,
            temperature=0.3,
            output_format="memory"
        )
        
        return response.text
    
    def generate_reflection(self,
                             user_memory: str,
                             item_memory: str,
                             decision: str,
                             ground_truth: str,
                             explanation: str,
                             graph_context: Optional[str] = None) -> Dict:
        """
        Generate collaborative reflection for memory update.
        
        Args:
            user_memory: User agent's current memory
            item_memory: Item agent's memory
            decision: Agent's decision ('positive'/'negative')
            ground_truth: Actual outcome
            explanation: Agent's explanation for decision
            graph_context: Optional graph context
        
        Returns:
            Reflection result dictionary
        """
        system_prompt = self.SYSTEM_PROMPTS['reflection']
        
        prompt = f"""Reflect on an incorrect recommendation decision.

YOUR CURRENT MEMORY:
{user_memory}

ITEM CONSIDERED:
{item_memory}

YOUR DECISION: {decision}
ACTUAL OUTCOME: {ground_truth}
YOUR EXPLANATION: {explanation}

{f"GRAPH CONTEXT:\n{graph_context}" if graph_context else ""}

Analyze and respond with JSON:
{{
    "analysis": "Why the decision was wrong",
    "missed_patterns": ["pattern1", "pattern2"],
    "new_preferences": ["new_pref1"],
    "new_dislikes": ["new_disl1"],
    "confidence_adjustment": -0.1,
    "item_insight": "What was misunderstood about the item"
}}"""
        
        return self.generate_with_json_output(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2
        )
    
    def fuse_memories(self,
                       current_memory: str,
                       new_perspectives: List[str],
                       max_words: int = 180) -> str:
        """
        Fuse multiple memory perspectives into a coherent whole.
        
        Args:
            current_memory: Current memory text
            new_perspectives: New perspectives to incorporate
            max_words: Maximum words in fused memory
        
        Returns:
            Fused memory text
        """
        system_prompt = self.SYSTEM_PROMPTS['fusion']
        
        perspectives_text = "\n".join(
            f"{i+1}. {p}" for i, p in enumerate(new_perspectives)
        )
        
        prompt = f"""Fuse the following perspectives into a coherent memory.

CURRENT MEMORY:
{current_memory}

NEW PERSPECTIVES:
{perspectives_text}

Requirements:
1. Keep total under {max_words} words
2. Incorporate the most important new information
3. Resolve contradictions logically
4. Maintain consistency with established patterns
5. Remove outdated or superseded information

Fused memory:"""
        
        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=512,
            temperature=0.3,
            output_format="memory"
        )
        
        return response.text
    
    def propagate_signals(self,
                           agent_memory: str,
                           neighbor_signals: List[Dict],
                           propagation_paths: List[str]) -> Dict:
        """
        Process neighborhood propagation signals.
        
        Args:
            agent_memory: Agent's current memory
            neighbor_signals: Signals from neighbor agents
            propagation_paths: Path descriptions
        
        Returns:
            Propagation result dictionary
        """
        system_prompt = self.SYSTEM_PROMPTS['propagation']
        
        signals_text = "\n".join(
            f"From {s.get('source', 'neighbor')} (strength: {s.get('strength', 0):.2f}): "
            f"{s.get('content', '')}"
            for s in neighbor_signals[:5]
        )
        
        paths_text = "\n".join(f"- {p}" for p in propagation_paths[:3])
        
        prompt = f"""Update your memory based on signals from similar agents.

YOUR MEMORY:
{agent_memory}

NEIGHBOR SIGNALS:
{signals_text}

PROPAGATION PATHS:
{paths_text}

Respond with JSON:
{{
    "adopted_patterns": ["pattern1"],
    "strengthened_patterns": ["pattern2"],
    "weakened_patterns": ["pattern3"],
    "rejected_patterns": ["pattern4"],
    "confidence_update": 0.0,
    "propagation_summary": "Brief summary"
}}"""
        
        return self.generate_with_json_output(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.2
        )
    
    def generate_batch(self,
                        prompts: List[str],
                        system_prompt: Optional[str] = None,
                        max_new_tokens: int = 256,
                        temperature: float = 0.3,
                        batch_size: int = 4) -> List[LocalLLMResponse]:
        """
        Generate responses for multiple prompts in batches.
        
        Args:
            prompts: List of prompt strings
            system_prompt: System prompt for all
            max_new_tokens: Maximum tokens per response
            temperature: Sampling temperature
            batch_size: Batch size for processing
        
        Returns:
            List of LocalLLMResponse objects
        """
        responses = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Build full prompts
            full_prompts = []
            for prompt in batch_prompts:
                full_prompts.append(
                    self._build_chat_prompt(
                        [{'role': 'user', 'content': prompt}],
                        system_prompt
                    )
                )
            
            # Tokenize batch
            inputs = self.tokenizer(
                full_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=8192 - max_new_tokens
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate batch
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            generation_time = time.time() / len(batch_prompts)
            
            # Decode responses
            for j, output_ids in enumerate(outputs):
                input_length = inputs['input_ids'][j].shape[0]
                generated_ids = output_ids[input_length:]
                generated_text = self.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                responses.append(LocalLLMResponse(
                    text=generated_text.strip(),
                    tokens_generated=len(generated_ids),
                    total_tokens=len(output_ids),
                    generation_time=generation_time,
                    tokens_per_second=len(generated_ids) / generation_time if generation_time > 0 else 0,
                    model_name=self.model_variant,
                    finish_reason='stop'
                ))
        
        return responses
    
    def clear_cache(self):
        """Clear model KV cache and conversation cache."""
        if hasattr(self.model, 'clear_cache'):
            self.model.clear_cache()
        
        # Clear conversation history
        self._conversation_cache.clear()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.debug("Model cache cleared")
    
    def get_statistics(self) -> Dict:
        """Get usage statistics."""
        stats = dict(self.stats)
        stats['avg_tokens_per_second'] = (
            self.stats['total_tokens_generated'] / max(1, self.stats['total_time'])
        )
        return stats
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_name': self.model_variant,
            'device': self.device,
            'quantization': self.quantization,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            'vocab_size': len(self.tokenizer),
            'max_context_length': getattr(self.model.config, 'max_position_embeddings', 8192),
            'memory_usage_mb': self.stats['peak_memory_mb']
        }
    
    def save_model(self, output_dir: str):
        """
        Save model and tokenizer.
        
        Args:
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config = {
            'model_variant': self.model_variant,
            'quantization': self.quantization,
            'device': self.device,
            'generation_config': self.generation_config.__dict__
        }
        
        with open(output_dir / 'llm_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Model saved successfully")


class Qwen3EmbeddingClient:
    """
    Embedding client using Qwen3 model for text-to-embedding conversion.
    Uses mean pooling of last hidden states.
    """
    
    def __init__(self,
                 llm_client: Qwen3LocalLLM,
                 embedding_dim: Optional[int] = None,
                 pooling_method: str = "mean"):
        """
        Initialize embedding client.
        
        Args:
            llm_client: Qwen3LocalLLM instance
            embedding_dim: Embedding dimension (auto-detected if None)
            pooling_method: Pooling method ('mean', 'cls', 'last')
        """
        self.llm_client = llm_client
        self.model = llm_client.model
        self.tokenizer = llm_client.tokenizer
        self.device = llm_client.device
        self.pooling_method = pooling_method
        
        # Detect embedding dimension
        if embedding_dim is None:
            self.embedding_dim = self.model.config.hidden_size
        else:
            self.embedding_dim = embedding_dim
    
    def encode(self, text: Union[str, List[str]], 
               batch_size: int = 8,
               normalize: bool = True) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: Single text or list of texts
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            Embedding tensor of shape (num_texts, embedding_dim)
        """
        if isinstance(text, str):
            text = [text]
        
        all_embeddings = []
        
        for i in range(0, len(text), batch_size):
            batch_texts = text[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get last hidden states
            hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            
            # Pool
            if self.pooling_method == "mean":
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
            elif self.pooling_method == "cls":
                embeddings = hidden_states[:, 0, :]
            elif self.pooling_method == "last":
                embeddings = hidden_states[:, -1, :]
            
            # Normalize
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity in [0, 1]
        """
        embeddings = self.encode([text1, text2])
        sim = F.cosine_similarity(
            embeddings[0:1], embeddings[1:2]
        )
        return max(0.0, float(sim))


# Factory function for quick initialization
def create_qwen3_client(
    model_size: str = "4b",
    quantization: str = "4bit",
    device: str = "auto",
    **kwargs
) -> Qwen3LocalLLM:
    """
    Factory function to quickly create a Qwen3 client.
    
    Args:
        model_size: Model size ('0.6b', '1.7b', '4b', '8b', '14b', '32b', '72b')
        quantization: Quantization method
        device: Device for inference
        **kwargs: Additional arguments for Qwen3LocalLLM
    
    Returns:
        Configured Qwen3LocalLLM instance
    """
    model_map = {
        '0.6b': Qwen3ModelVariant.QWEN3_0_6B,
        '1.7b': Qwen3ModelVariant.QWEN3_1_7B_INSTRUCT,
        '4b': Qwen3ModelVariant.QWEN3_4B_INSTRUCT,
        '8b': Qwen3ModelVariant.QWEN3_8B_INSTRUCT,
        '14b': Qwen3ModelVariant.QWEN3_14B_INSTRUCT,
        '32b': Qwen3ModelVariant.QWEN3_32B_INSTRUCT,
        '72b': Qwen3ModelVariant.QWEN3_72B_INSTRUCT,
    }
    
    variant = model_map.get(model_size.lower(), Qwen3ModelVariant.QWEN3_4B_INSTRUCT)
    
    return Qwen3LocalLLM(
        model_variant=variant,
        quantization=quantization,
        device=device,
        **kwargs
    )