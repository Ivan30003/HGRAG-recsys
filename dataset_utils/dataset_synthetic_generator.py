"""
Synthetic Data Generator Module
Generates synthetic recommendation data for testing and development.
Useful when Amazon data is not available or for controlled experiments.
"""

import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generates synthetic user-item interaction data for testing.
    
    Creates realistic patterns including:
    - User preference clusters
    - Item categories and attributes
    - Rating patterns
    - Temporal dynamics
    """
    
    # Item categories with realistic attributes
    CATEGORIES = {
        'Music': {
            'genres': ['Rock', 'Jazz', 'Classical', 'Pop', 'Electronic', 'Hip-Hop', 'Country', 'Blues'],
            'formats': ['CD', 'Vinyl', 'Digital'],
            'eras': ['1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']
        },
        'Books': {
            'genres': ['Fiction', 'Non-Fiction', 'Science', 'History', 'Biography', 'Fantasy', 'Mystery', 'Romance'],
            'formats': ['Hardcover', 'Paperback', 'E-book', 'Audiobook'],
            'lengths': ['Short', 'Medium', 'Long']
        },
        'Electronics': {
            'types': ['Headphones', 'Speakers', 'Cameras', 'Laptops', 'Tablets', 'Smartphones', 'Accessories'],
            'brands': ['TechPro', 'AudioMax', 'VisionPlus', 'ComputeX', 'SmartLife'],
            'price_tiers': ['Budget', 'Mid-range', 'Premium', 'Professional']
        },
        'Office': {
            'types': ['Desk', 'Chair', 'Storage', 'Lighting', 'Stationery', 'Organization', 'Technology'],
            'brands': ['OfficePro', 'WorkSmart', 'ErgoPlus', 'BasicLine', 'PremiumOffice'],
            'styles': ['Modern', 'Traditional', 'Minimalist', 'Industrial']
        }
    }
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        self.random_seed = random_seed
    
    def generate_dataset(self,
                          num_users: int = 500,
                          num_items: int = 1000,
                          avg_interactions_per_user: int = 15,
                          category: str = 'Music',
                          rating_noise: float = 0.2,
                          temporal_spread_days: int = 365) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_users: Number of users
            num_items: Number of items
            avg_interactions_per_user: Average interactions per user
            category: Product category
            rating_noise: Noise level in ratings
            temporal_spread_days: Days over which interactions occur
        
        Returns:
            Tuple of (reviews DataFrame, item metadata dict)
        """
        logger.info(f"Generating synthetic dataset: {num_users} users, "
                   f"{num_items} items, category={category}")
        
        # Generate items
        items = self._generate_items(num_items, category)
        
        # Generate user preferences
        user_preferences = self._generate_user_preferences(num_users, category)
        
        # Generate user clusters for collaborative patterns
        user_clusters = self._generate_user_clusters(num_users, n_clusters=10)
        
        # Generate interactions
        reviews = self._generate_interactions(
            user_preferences, items, user_clusters,
            avg_interactions_per_user, rating_noise,
            temporal_spread_days
        )
        
        logger.info(f"Generated {len(reviews)} interactions")
        
        return reviews, items
    
    def _generate_items(self, num_items: int, category: str) -> Dict[str, Dict]:
        """
        Generate synthetic items with realistic attributes.
        
        Args:
            num_items: Number of items
            category: Product category
        
        Returns:
            Dictionary of item metadata
        """
        cat_config = self.CATEGORIES.get(category, self.CATEGORIES['Music'])
        items = {}
        
        for i in range(num_items):
            asin = f'ITEM_{i:06d}'
            
            # Select attributes
            if 'genres' in cat_config:
                genre = np.random.choice(cat_config['genres'])
            elif 'types' in cat_config:
                genre = np.random.choice(cat_config['types'])
            else:
                genre = 'General'
            
            # Create title
            adjectives = ['Premium', 'Classic', 'Essential', 'Ultimate', 'Modern',
                         'Vintage', 'Professional', 'Deluxe', 'Basic', 'Advanced']
            title = f"{np.random.choice(adjectives)} {genre} Item {i+1}"
            
            # Description
            desc_parts = [
                f"This {genre.lower()} product offers exceptional quality.",
                f"Features include premium materials and expert craftsmanship.",
                f"Perfect for enthusiasts and professionals alike.",
                f"Comes with a satisfaction guarantee."
            ]
            description = ' '.join(np.random.choice(desc_parts, size=2, replace=False))
            
            items[asin] = {
                'asin': asin,
                'title': title,
                'description': description,
                'main_category': category,
                'sub_category': genre,
                'brand': np.random.choice(cat_config.get('brands', ['BrandA', 'BrandB'])),
                'price': round(np.random.uniform(9.99, 299.99), 2),
                'avg_rating': round(np.random.uniform(3.0, 5.0), 1)
            }
        
        return items
    
    def _generate_user_preferences(self, 
                                     num_users: int, 
                                     category: str) -> Dict[str, Dict]:
        """
        Generate synthetic user preferences.
        
        Args:
            num_users: Number of users
            category: Product category
        
        Returns:
            Dictionary of user preferences
        """
        cat_config = self.CATEGORIES.get(category, self.CATEGORIES['Music'])
        
        if 'genres' in cat_config:
            attr_key = 'genres'
        elif 'types' in cat_config:
            attr_key = 'types'
        else:
            attr_key = list(cat_config.keys())[0]
        
        attributes = cat_config.get(attr_key, ['General'])
        
        preferences = {}
        
        for i in range(num_users):
            user_id = f'USER_{i:06d}'
            
            # Each user prefers 2-4 categories
            n_prefs = np.random.randint(2, 5)
            preferred = list(np.random.choice(attributes, size=min(n_prefs, len(attributes)), replace=False))
            
            # Each user dislikes 1-2 categories
            remaining = [a for a in attributes if a not in preferred]
            n_dislikes = min(np.random.randint(1, 3), len(remaining))
            disliked = list(np.random.choice(remaining, size=n_dislikes, replace=False)) if remaining else []
            
            preferences[user_id] = {
                'user_id': user_id,
                'preferred_categories': preferred,
                'disliked_categories': disliked,
                'rating_tendency': np.random.choice(['generous', 'neutral', 'critical'], p=[0.3, 0.5, 0.2]),
                'activity_level': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            }
        
        return preferences
    
    def _generate_user_clusters(self, 
                                  num_users: int, 
                                  n_clusters: int = 10) -> Dict[str, int]:
        """
        Generate user clusters for collaborative patterns.
        
        Args:
            num_users: Number of users
            n_clusters: Number of clusters
        
        Returns:
            Dictionary mapping user_id -> cluster_id
        """
        clusters = {}
        
        for i in range(num_users):
            user_id = f'USER_{i:06d}'
            cluster_id = i % n_clusters
            clusters[user_id] = cluster_id
        
        return clusters
    
    def _generate_interactions(self,
                                user_preferences: Dict,
                                items: Dict,
                                user_clusters: Dict,
                                avg_interactions: int,
                                noise: float,
                                temporal_spread_days: int) -> pd.DataFrame:
        """
        Generate synthetic interactions with realistic patterns.
        
        Args:
            user_preferences: User preference dict
            items: Item metadata dict
            user_clusters: User cluster assignments
            avg_interactions: Average interactions per user
            noise: Rating noise level
            temporal_spread_days: Temporal spread in days
        
        Returns:
            DataFrame of interactions
        """
        interactions = []
        base_time = datetime.now() - timedelta(days=temporal_spread_days)
        
        item_ids = list(items.keys())
        
        for user_id, prefs in user_preferences.items():
            # Determine number of interactions
            activity_multiplier = {
                'low': 0.5, 'medium': 1.0, 'high': 2.0
            }[prefs['activity_level']]
            
            n_interactions = max(5, int(np.random.poisson(avg_interactions * activity_multiplier)))
            
            # Generate interaction timestamps
            timestamps = sorted([
                base_time + timedelta(
                    days=np.random.randint(0, temporal_spread_days),
                    hours=np.random.randint(0, 24)
                )
                for _ in range(n_interactions)
            ])
            
            for ts in timestamps:
                # Select item based on preferences
                preferred_items = [
                    iid for iid, item in items.items()
                    if item.get('sub_category') in prefs['preferred_categories']
                ]
                
                disliked_items = [
                    iid for iid, item in items.items()
                    if item.get('sub_category') in prefs['disliked_categories']
                ]
                
                # 70% chance of picking preferred item
                if preferred_items and np.random.random() < 0.7:
                    item_id = np.random.choice(preferred_items)
                    base_rating = np.random.uniform(3.5, 5.0)
                elif disliked_items and np.random.random() < 0.1:
                    item_id = np.random.choice(disliked_items)
                    base_rating = np.random.uniform(1.0, 3.0)
                else:
                    item_id = np.random.choice(item_ids)
                    base_rating = np.random.uniform(2.5, 4.5)
                
                # Add noise
                rating = base_rating + np.random.normal(0, noise)
                rating = max(1.0, min(5.0, rating))
                
                # Adjust for rating tendency
                tendency_multiplier = {
                    'generous': 0.3,
                    'neutral': 0.0,
                    'critical': -0.3
                }[prefs['rating_tendency']]
                rating = max(1.0, min(5.0, rating + tendency_multiplier))
                
                # Generate review text
                review_text = self._generate_review_text(
                    items[item_id], rating, prefs
                )
                
                interactions.append({
                    'reviewerID': user_id,
                    'asin': item_id,
                    'overall': round(rating, 1),
                    'reviewText': review_text,
                    'summary': review_text[:100],
                    'unixReviewTime': int(ts.timestamp()),
                    'verified': np.random.random() < 0.8,
                    'vote': max(0, int(np.random.exponential(2)))
                })
        
        return pd.DataFrame(interactions)
    
    def _generate_review_text(self,
                               item: Dict,
                               rating: float,
                               user_prefs: Dict) -> str:
        """
        Generate synthetic review text.
        
        Args:
            item: Item metadata
            rating: Rating value
            user_prefs: User preferences
        
        Returns:
            Synthetic review text
        """
        positive_templates = [
            "I really enjoyed this {category} product. The {sub_category} aspects were exactly what I was looking for. Highly recommended!",
            "Great {category} item. The quality exceeded my expectations. Would definitely buy again.",
            "Excellent {sub_category} product. Perfect for anyone who appreciates good {category}. Five stars!",
            "Love this! As a fan of {sub_category}, this really hit the mark. The craftsmanship is superb."
        ]
        
        neutral_templates = [
            "Decent {category} product. It's good for the price, but nothing extraordinary.",
            "Ok {sub_category} item. Does the job, but I've seen better. Average overall.",
            "Standard {category}. Meets expectations but doesn't exceed them. Solid choice."
        ]
        
        negative_templates = [
            "Disappointed with this {category} product. The {sub_category} quality was below expectations.",
            "Not what I hoped for. As someone who usually enjoys {sub_category}, this was underwhelming.",
            "Below average {category} item. Would not recommend to others. Look elsewhere."
        ]
        
        if rating >= 4.0:
            template = np.random.choice(positive_templates)
        elif rating >= 3.0:
            template = np.random.choice(neutral_templates)
        else:
            template = np.random.choice(negative_templates)
        
        return template.format(
            category=item.get('main_category', 'product'),
            sub_category=item.get('sub_category', 'general')
        )