"""
Intrinsic Memory Module
Handles immutable agent features that serve as identity anchors.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import hashlib
import json


@dataclass
class IntrinsicMemory:
    """
    Immutable memory tier capturing core identity features.
    Frozen after initialization to prevent concept drift.
    """
    
    # Core identity fields
    agent_id: str
    agent_type: str  # 'user' or 'item'
    
    # Content features (item-specific)
    title: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    brand: Optional[str] = None
    
    # Preference features (user-specific)
    demographic_info: Optional[Dict] = None
    explicit_preferences: List[str] = field(default_factory=list)
    stated_constraints: List[str] = field(default_factory=list)
    
    # Encoded representation (set once, never modified)
    text_embedding: Optional[List[float]] = None
    summary_text: Optional[str] = None
    
    def __post_init__(self):
        """Validate and freeze the memory after initialization."""
        self._frozen = True
        self._compute_hash()
    
    def _compute_hash(self):
        """Compute a unique hash for this memory state."""
        content = f"{self.agent_id}_{self.agent_type}_{self.title}_{self.category}"
        self.content_hash = hashlib.md5(content.encode()).hexdigest()
    
    def set_embedding(self, embedding: List[float]):
        """Set the text embedding (called once during initialization)."""
        if hasattr(self, 'text_embedding') and self.text_embedding is not None:
            raise ValueError("Intrinsic memory embedding is immutable once set")
        self.text_embedding = embedding
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'title': self.title,
            'category': self.category,
            'description': self.description,
            'brand': self.brand,
            'demographic_info': self.demographic_info,
            'explicit_preferences': self.explicit_preferences,
            'stated_constraints': self.stated_constraints,
            'summary_text': self.summary_text,
            'content_hash': self.content_hash
        }
    
    def to_prompt_text(self) -> str:
        """Generate text representation for LLM prompts."""
        if self.agent_type == 'item':
            parts = []
            if self.title:
                parts.append(f"Title: {self.title}")
            if self.category:
                parts.append(f"Category: {self.category}")
            if self.description:
                parts.append(f"Description: {self.description}")
            if self.brand:
                parts.append(f"Brand: {self.brand}")
            return "\n".join(parts) if parts else f"Item: {self.agent_id}"
        
        elif self.agent_type == 'user':
            parts = []
            if self.explicit_preferences:
                parts.append(f"Explicit preferences: {', '.join(self.explicit_preferences)}")
            if self.stated_constraints:
                parts.append(f"Constraints: {', '.join(self.stated_constraints)}")
            if self.demographic_info:
                parts.append(f"Demographics: {json.dumps(self.demographic_info)}")
            return "\n".join(parts) if parts else f"User: {self.agent_id}"
        
        return f"Agent: {self.agent_id}"
    
    def __setattr__(self, name, value):
        """Prevent modification of frozen memory."""
        if hasattr(self, '_frozen') and self._frozen and name not in ['_frozen']:
            if name in ['text_embedding', 'summary_text']:
                # Allow setting these once
                if getattr(self, name, None) is not None:
                    raise AttributeError(f"Cannot modify frozen intrinsic memory: {name}")
            elif name != 'content_hash' and not name.startswith('_'):
                raise AttributeError(f"Cannot modify frozen intrinsic memory: {name}")
        super().__setattr__(name, value)


def create_intrinsic_memory_from_item(item_data: Dict, item_id: str) -> IntrinsicMemory:
    """Factory function to create intrinsic memory from item data."""
    return IntrinsicMemory(
        agent_id=item_id,
        agent_type='item',
        title=item_data.get('title', ''),
        category=item_data.get('category', ''),
        description=item_data.get('description', ''),
        brand=item_data.get('brand', None)
    )


def create_intrinsic_memory_from_user(user_data: Dict, user_id: str) -> IntrinsicMemory:
    """Factory function to create intrinsic memory from user data."""
    return IntrinsicMemory(
        agent_id=user_id,
        agent_type='user',
        explicit_preferences=user_data.get('preferences', []),
        stated_constraints=user_data.get('constraints', []),
        demographic_info=user_data.get('demographics', None)
    )