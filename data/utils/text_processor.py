"""
text_processor.py - Advanced text processing utilities for H-GRAGrecsys

This module provides comprehensive text processing capabilities including
cleaning, tokenization, summarization, keyword extraction, and embeddings
for both item descriptions and user preferences.
"""

import re
import hashlib
import json
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
import spacy
from tqdm import tqdm
import logging
from dataclasses import dataclass, field

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class TextProcessingResult:
    """Container for text processing results."""
    original: str
    cleaned: str
    tokens: List[str]
    lemmatized: List[str]
    stemmed: List[str]
    keywords: List[str]
    summary: str
    embedding: Optional[np.ndarray] = None
    ner_entities: Optional[List[Dict]] = None
    pos_tags: Optional[List[Tuple[str, str]]] = None
    word_frequencies: Optional[Dict[str, int]] = None
    sentence_count: int = 0
    word_count: int = 0
    avg_word_length: float = 0.0


class TextCleaner:
    """Text cleaning utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TextCleaner.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Compile regex patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.emoji_pattern = re.compile(r'[^\w\s.,!?;:\'"]')
        self.multiple_spaces = re.compile(r'\s+')
        self.newline_pattern = re.compile(r'\n+')
        
        # Punctuation to keep
        self.punct_keep = set(".,!?;:'\"")
        
        # Common abbreviations to preserve
        self.abbreviations = set([
            'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'rev.', 'hon.', 
            'st.', 'ave.', 'blvd.', 'rd.', 'ln.', 'apt.', 'ste.',
            'etc.', 'e.g.', 'i.e.', 'vs.', 'inc.', 'ltd.', 'co.'
        ])
        
        # Custom stopwords
        self.custom_stopwords = set(config.get('custom_stopwords', []))
        self.keep_words = set(config.get('keep_words', []))
        
        logger.info("TextCleaner initialized")
    
    def clean(self, text: str, remove_stopwords: bool = False, 
             lowercase: bool = True) -> str:
        """
        Clean text with multiple processing steps.
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            lowercase: Whether to convert to lowercase
        
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Initial cleaning
        text = self._remove_html(text)
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = self._normalize_whitespace(text)
        text = self._remove_emojis(text)
        
        # Handle case
        if lowercase:
            text = text.lower()
        
        # Remove stopwords if requested
        if remove_stopwords:
            text = self._remove_stopwords(text)
        
        # Clean punctuation
        text = self._clean_punctuation(text)
        
        # Normalize again
        text = self._normalize_whitespace(text)
        
        return text.strip()
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags."""
        return self.html_pattern.sub('', text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs."""
        return self.url_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses."""
        return self.email_pattern.sub('', text)
    
    def _remove_emojis(self, text: str) -> str:
        """Remove emojis and special characters."""
        return self.emoji_pattern.sub('', text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace."""
        text = self.newline_pattern.sub(' ', text)
        text = self.multiple_spaces.sub(' ', text)
        return text.strip()
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        if not text:
            return ""
        
        # Get stopwords
        stopwords_set = set(stopwords.words('english'))
        stopwords_set.update(self.custom_stopwords)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Keep only non-stopwords
        filtered = [t for t in tokens if t.lower() not in stopwords_set or t.lower() in self.keep_words]
        
        return ' '.join(filtered)
    
    def _clean_punctuation(self, text: str) -> str:
        """Clean punctuation while preserving sentence structure."""
        if not text:
            return ""
        
        # Replace abbreviations
        for abbr in self.abbreviations:
            text = text.replace(abbr, abbr.replace('.', '__DOT__'))
        
        # Remove unwanted punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"]', ' ', text)
        
        # Restore abbreviations
        text = text.replace('__DOT__', '.')
        
        return text
    
    def is_valid_text(self, text: str, min_length: int = 10) -> bool:
        """
        Check if text is valid.
        
        Args:
            text: Text to check
            min_length: Minimum length
        
        Returns:
            True if valid
        """
        if not text or not isinstance(text, str):
            return False
        
        cleaned = self.clean(text)
        
        # Check length
        if len(cleaned.split()) < min_length // 10:
            return False
        
        # Check if it's mostly numbers or special characters
        alpha_ratio = sum(c.isalpha() for c in cleaned) / max(len(cleaned), 1)
        if alpha_ratio < 0.3:
            return False
        
        return True


class TextTokenizer:
    """Text tokenization utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TextTokenizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize stemmer and lemmatizer
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize spaCy if available
        self.use_spacy = config.get('use_spacy', False)
        self.nlp = None
        
        if self.use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("spaCy model not found. Falling back to NLTK.")
                self.use_spacy = False
        
        logger.info("TextTokenizer initialized")
    
    def tokenize(self, text: str, method: str = 'word') -> List[str]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            method: 'word', 'sentence', or 'char'
        
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        if method == 'sentence':
            return sent_tokenize(text)
        elif method == 'char':
            return list(text)
        else:  # word tokenization
            if self.use_spacy and self.nlp:
                doc = self.nlp(text)
                return [token.text for token in doc]
            else:
                return word_tokenize(text)
    
    def tokenize_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """
        Tokenize with POS tagging.
        
        Args:
            text: Input text
        
        Returns:
            List of (token, pos_tag) tuples
        """
        if not text:
            return []
        
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            return [(token.text, token.pos_) for token in doc]
        else:
            tokens = word_tokenize(text)
            return pos_tag(tokens)
    
    def stem(self, tokens: List[str]) -> List[str]:
        """
        Apply stemming to tokens.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens: List of tokens
        
        Returns:
            Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def get_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        return sent_tokenize(text)


class KeywordExtractor:
    """Keyword extraction utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize KeywordExtractor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=config.get('max_keyword_features', 10000),
            stop_words='english',
            max_df=0.8,
            min_df=2
        )
        
        # Parameters
        self.max_keywords = config.get('max_keywords', 15)
        self.min_keyword_length = config.get('min_keyword_length', 3)
        
        logger.info("KeywordExtractor initialized")
    
    def extract_keywords(self, text: str, top_k: int = 10, 
                        method: str = 'tfidf') -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            top_k: Number of keywords to extract
            method: 'tfidf', 'frequency', 'rake', or 'textrank'
        
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        cleaned_text = self._clean_text(text)
        
        if method == 'tfidf':
            return self._extract_tfidf_keywords(cleaned_text, top_k)
        elif method == 'frequency':
            return self._extract_frequency_keywords(cleaned_text, top_k)
        elif method == 'rake':
            return self._extract_rake_keywords(cleaned_text, top_k)
        elif method == 'textrank':
            return self._extract_textrank_keywords(cleaned_text, top_k)
        else:
            raise ValueError(f"Unknown keyword extraction method: {method}")
    
    def _clean_text(self, text: str) -> str:
        """Clean text for keyword extraction."""
        # Remove stopwords and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _extract_tfidf_keywords(self, text: str, top_k: int) -> List[str]:
        """Extract keywords using TF-IDF."""
        # Tokenize
        tokens = word_tokenize(text)
        tokens = [t.lower() for t in tokens if t.isalpha() and len(t) >= self.min_keyword_length]
        
        if not tokens:
            return []
        
        # Calculate term frequencies
        freq = Counter(tokens)
        
        # Filter common words
        stopwords_set = set(stopwords.words('english'))
        filtered = {word: freq for word, freq in freq.items() 
                   if word not in stopwords_set and len(word) >= self.min_keyword_length}
        
        # Sort by frequency
        keywords = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in keywords[:top_k]]
    
    def _extract_frequency_keywords(self, text: str, top_k: int) -> List[str]:
        """Extract keywords by frequency."""
        tokens = word_tokenize(text)
        tokens = [t.lower() for t in tokens if t.isalpha() and len(t) >= self.min_keyword_length]
        
        # Remove stopwords
        stopwords_set = set(stopwords.words('english'))
        tokens = [t for t in tokens if t not in stopwords_set]
        
        # Count frequencies
        freq = Counter(tokens)
        
        # Get top k
        keywords = [word for word, _ in freq.most_common(top_k)]
        
        return keywords
    
    def _extract_rake_keywords(self, text: str, top_k: int) -> List[str]:
        """Extract keywords using RAKE algorithm."""
        # Simple RAKE implementation
        stopwords_set = set(stopwords.words('english'))
        
        # Tokenize and split into phrases
        sentences = sent_tokenize(text)
        phrases = []
        
        for sentence in sentences:
            # Split by stopwords
            words = word_tokenize(sentence.lower())
            phrase = []
            
            for word in words:
                if word.isalpha():
                    if word in stopwords_set or len(word) < self.min_keyword_length:
                        if phrase:
                            phrases.append(' '.join(phrase))
                            phrase = []
                    else:
                        phrase.append(word)
            
            if phrase:
                phrases.append(' '.join(phrase))
        
        # Score phrases
        phrase_freq = Counter(phrases)
        word_freq = Counter()
        word_degree = Counter()
        
        for phrase in phrases:
            words = phrase.split()
            for word in words:
                word_freq[word] += 1
                word_degree[word] += len(words)
        
        # Calculate RAKE scores
        phrase_scores = {}
        for phrase, freq in phrase_freq.items():
            words = phrase.split()
            score = sum(word_degree[word] / word_freq[word] for word in words)
            phrase_scores[phrase] = score * freq
        
        # Sort and return
        keywords = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in keywords[:top_k]]
    
    def _extract_textrank_keywords(self, text: str, top_k: int) -> List[str]:
        """Extract keywords using TextRank algorithm."""
        # Simplified TextRank implementation
        # For production, use sumy or pytextrank
        return self._extract_frequency_keywords(text, top_k)
    
    def extract_keyphrases(self, text: str, top_k: int = 5) -> List[str]:
        """
        Extract keyphrases from text.
        
        Args:
            text: Input text
            top_k: Number of keyphrases
        
        Returns:
            List of keyphrases
        """
        return self.extract_keywords(text, top_k, method='rake')


class TextSummarizer:
    """Text summarization utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TextSummarizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Parameters
        self.max_summary_length = config.get('max_summary_length', 50)
        self.min_summary_length = config.get('min_summary_length', 10)
        
        # Initialize summarization model if available
        self.use_transformer = config.get('use_transformer_summarization', False)
        self.summarizer = None
        
        if self.use_transformer:
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model=config.get('summarization_model', 'facebook/bart-large-cnn')
                )
                logger.info("Transformer summarization model loaded")
            except Exception as e:
                logger.warning(f"Failed to load summarization model: {e}")
                self.use_transformer = False
        
        # Initialize TF-IDF vectorizer for extractive summarization
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )
        
        logger.info("TextSummarizer initialized")
    
    def summarize(self, text: str, max_words: Optional[int] = None,
                 method: str = 'extractive') -> str:
        """
        Generate summary of text.
        
        Args:
            text: Input text
            max_words: Maximum words in summary
            method: 'extractive', 'abstractive', or 'hybrid'
        
        Returns:
            Summary string
        """
        if not text:
            return ""
        
        if max_words is None:
            max_words = self.max_summary_length
        
        if method == 'abstractive' and self.use_transformer:
            return self._abstractive_summarize(text, max_words)
        elif method == 'hybrid':
            return self._hybrid_summarize(text, max_words)
        else:
            return self._extractive_summarize(text, max_words)
    
    def _extractive_summarize(self, text: str, max_words: int) -> str:
        """Generate extractive summary."""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= 1:
            # Truncate if only one sentence
            words = text.split()
            if len(words) > max_words:
                return ' '.join(words[:max_words]) + '...'
            return text
        
        # Score sentences using TF-IDF
        try:
            # Filter sentences
            valid_sentences = [s for s in sentences if len(s.split()) > 3]
            
            if not valid_sentences:
                return sentences[0][:max_words * 10]
            
            # Calculate TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_sentences)
            scores = tfidf_matrix.sum(axis=1).A1
            
            # Rank sentences
            scored_sentences = list(zip(valid_sentences, scores))
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            
            # Select sentences
            selected = []
            current_words = 0
            
            # Always include first sentence
            if valid_sentences[0] != scored_sentences[0][0]:
                selected.append(valid_sentences[0])
                current_words += len(valid_sentences[0].split())
            
            # Add top scoring sentences
            for sentence, _ in scored_sentences:
                sentence_words = len(sentence.split())
                if current_words + sentence_words <= max_words and sentence not in selected:
                    selected.append(sentence)
                    current_words += sentence_words
                
                if current_words >= max_words:
                    break
            
            # If no sentences selected, take first
            if not selected:
                return sentences[0][:max_words * 10]
            
            # Order sentences by original position
            selected_order = []
            for sentence in sentences:
                if sentence in selected:
                    selected_order.append(sentence)
            
            summary = ' '.join(selected_order)
            
        except Exception as e:
            logger.warning(f"Extractive summarization failed: {e}")
            # Fallback: take first few sentences
            summary = ' '.join(sentences[:3])
        
        # Truncate if still too long
        words = summary.split()
        if len(words) > max_words:
            summary = ' '.join(words[:max_words]) + '...'
        
        return summary
    
    def _abstractive_summarize(self, text: str, max_words: int) -> str:
        """Generate abstractive summary."""
        if not self.summarizer:
            return self._extractive_summarize(text, max_words)
        
        try:
            # Calculate max/min lengths for summarizer
            input_length = len(text.split())
            max_len = min(max_words * 2, 200)
            min_len = min(max_words // 2, 30)
            
            # Generate summary
            result = self.summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )
            
            summary = result[0]['summary_text']
            
            # Truncate if needed
            words = summary.split()
            if len(words) > max_words:
                summary = ' '.join(words[:max_words]) + '...'
            
            return summary
            
        except Exception as e:
            logger.warning(f"Abstractive summarization failed: {e}")
            return self._extractive_summarize(text, max_words)
    
    def _hybrid_summarize(self, text: str, max_words: int) -> str:
        """Generate hybrid summary."""
        # Try abstractive first
        if self.use_transformer:
            try:
                abstractive = self._abstractive_summarize(text, max_words)
                if len(abstractive.split()) > 5:
                    return abstractive
            except:
                pass
        
        # Fallback to extractive
        return self._extractive_summarize(text, max_words)
    
    def summarize_sentences(self, sentences: List[str], max_words: int) -> List[str]:
        """
        Summarize a list of sentences.
        
        Args:
            sentences: List of sentences
            max_words: Maximum words in summary
        
        Returns:
            List of summarized sentences
        """
        text = ' '.join(sentences)
        summary = self.summarize(text, max_words)
        return sent_tokenize(summary)


class TextEmbedder:
    """Text embedding utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TextEmbedder.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Embedding parameters
        self.embedding_dim = config.get('embedding_dim', 384)
        self.embedding_method = config.get('embedding_method', 'sentence-transformers')
        
        # Initialize models
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.embedding_method == 'sentence-transformers':
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(
                    config.get('sentence_transformer_model', 'all-MiniLM-L6-v2'),
                    device=self.device
                )
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info("SentenceTransformer model loaded")
            except ImportError:
                logger.warning("sentence-transformers not available")
                self.embedding_method = 'transformers'
        
        if self.embedding_method == 'transformers':
            try:
                model_name = config.get('transformer_model', 'bert-base-uncased')
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                self.embedding_dim = self.model.config.hidden_size
                logger.info("Transformer model loaded")
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}")
                self.embedding_method = 'tfidf'
        
        if self.embedding_method == 'tfidf':
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=config.get('tfidf_features', 10000),
                stop_words='english'
            )
            self.embedding_dim = 10000
            self.pca = PCA(n_components=min(self.embedding_dim, 256))
            logger.info("Using TF-IDF embeddings")
        
        logger.info(f"TextEmbedder initialized with dim={self.embedding_dim}")
    
    def embed(self, text: str, normalize_embedding: bool = True) -> np.ndarray:
        """
        Generate text embedding.
        
        Args:
            text: Input text
            normalize_embedding: Whether to normalize
        
        Returns:
            Embedding vector
        """
        if not text:
            return np.zeros(self.embedding_dim)
        
        if self.embedding_method == 'sentence-transformers':
            embedding = self.model.encode(text, normalize_embeddings=True)
            
        elif self.embedding_method == 'transformers':
            embedding = self._embed_transformers(text)
            
        elif self.embedding_method == 'tfidf':
            embedding = self._embed_tfidf(text)
            
        else:
            embedding = self._embed_hash(text)
        
        # Normalize if needed
        if normalize_embedding and len(embedding) > 0:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def embed_batch(self, texts: List[str], normalize_embedding: bool = True,
                   batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            normalize_embedding: Whether to normalize
            batch_size: Batch size for processing
        
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        
        if self.embedding_method == 'sentence-transformers':
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=normalize_embedding,
                batch_size=batch_size
            )
            
        else:
            for text in tqdm(texts, desc="Generating embeddings"):
                embedding = self.embed(text, normalize_embedding)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
        
        return embeddings
    
    def _embed_transformers(self, text: str) -> np.ndarray:
        """Generate embedding using transformers."""
        if not self.tokenizer or not self.model:
            return np.zeros(self.embedding_dim)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            logger.warning(f"Transformer embedding failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def _embed_tfidf(self, text: str) -> np.ndarray:
        """Generate embedding using TF-IDF."""
        try:
            # Fit if not fitted
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                # Fit on sample text
                self.tfidf_vectorizer.fit([text])
            
            # Transform
            embedding = self.tfidf_vectorizer.transform([text]).toarray().squeeze()
            
            # Reduce dimensionality if needed
            if len(embedding) > 256:
                if not hasattr(self.pca, 'components_'):
                    # Fit PCA on this single vector (not ideal but works)
                    self.pca.fit(embedding.reshape(1, -1))
                embedding = self.pca.transform(embedding.reshape(1, -1)).squeeze()
            
            return embedding
            
        except Exception as e:
            logger.warning(f"TF-IDF embedding failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def _embed_hash(self, text: str) -> np.ndarray:
        """Generate embedding using hash-based method."""
        # Simple hash-based embedding
        embedding = np.zeros(self.embedding_dim)
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        for token in tokens:
            if token.isalpha():
                # Hash token to create features
                hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
                idx = hash_val % self.embedding_dim
                embedding[idx] += 1
        
        return embedding


class TextProcessor:
    """Main text processing class combining all utilities."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TextProcessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.cleaner = TextCleaner(config)
        self.tokenizer = TextTokenizer(config)
        self.keyword_extractor = KeywordExtractor(config)
        self.summarizer = TextSummarizer(config)
        self.embedder = TextEmbedder(config)
        
        logger.info("TextProcessor initialized")
    
    def process_text(self, text: str, extract_keywords: bool = True,
                    generate_summary: bool = True, 
                    generate_embedding: bool = True) -> TextProcessingResult:
        """
        Process text through all pipeline stages.
        
        Args:
            text: Input text
            extract_keywords: Whether to extract keywords
            generate_summary: Whether to generate summary
            generate_embedding: Whether to generate embedding
        
        Returns:
            TextProcessingResult object
        """
        if not text:
            return TextProcessingResult(
                original="",
                cleaned="",
                tokens=[],
                lemmatized=[],
                stemmed=[],
                keywords=[],
                summary="",
                embedding=np.zeros(self.embedder.embedding_dim)
            )
        
        # Clean text
        cleaned = self.cleaner.clean(text)
        
        # Tokenize
        tokens = self.tokenizer.tokenize(cleaned)
        lemmatized = self.tokenizer.lemmatize(tokens)
        stemmed = self.tokenizer.stem(tokens)
        
        # Get POS tags
        pos_tags = self.tokenizer.tokenize_with_pos(cleaned) if cleaned else []
        
        # Extract keywords
        keywords = []
        if extract_keywords and cleaned:
            keywords = self.keyword_extractor.extract_keywords(
                cleaned, top_k=self.keyword_extractor.max_keywords
            )
        
        # Generate summary
        summary = ""
        if generate_summary and cleaned:
            summary = self.summarizer.summarize(cleaned)
        
        # Generate embedding
        embedding = None
        if generate_embedding and cleaned:
            embedding = self.embedder.embed(cleaned)
        
        # Word frequencies
        word_freq = Counter(tokens)
        
        # Create result
        result = TextProcessingResult(
            original=text,
            cleaned=cleaned,
            tokens=tokens,
            lemmatized=lemmatized,
            stemmed=stemmed,
            keywords=keywords,
            summary=summary,
            embedding=embedding,
            pos_tags=pos_tags,
            word_frequencies=dict(word_freq),
            sentence_count=len(sent_tokenize(text)),
            word_count=len(tokens),
            avg_word_length=np.mean([len(t) for t in tokens]) if tokens else 0
        )
        
        return result
    
    def process_texts(self, texts: List[str], **kwargs) -> List[TextProcessingResult]:
        """
        Process multiple texts.
        
        Args:
            texts: List of input texts
            **kwargs: Additional arguments for process_text
        
        Returns:
            List of TextProcessingResult objects
        """
        results = []
        for text in tqdm(texts, desc="Processing texts"):
            result = self.process_text(text, **kwargs)
            results.append(result)
        
        return results
    
    def compute_similarity(self, text1: str, text2: str, method: str = 'cosine') -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            method: 'cosine', 'jaccard', or 'euclidean'
        
        Returns:
            Similarity score
        """
        if not text1 or not text2:
            return 0.0
        
        # Get embeddings
        emb1 = self.embedder.embed(text1)
        emb2 = self.embedder.embed(text2)
        
        if len(emb1) != len(emb2):
            return 0.0
        
        # Compute similarity
        if method == 'cosine':
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            
        elif method == 'jaccard':
            # Use token sets
            tokens1 = set(self.tokenizer.tokenize(text1))
            tokens2 = set(self.tokenizer.tokenize(text2))
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            similarity = intersection / union if union > 0 else 0.0
            
        elif method == 'euclidean':
            distance = np.linalg.norm(emb1 - emb2)
            similarity = 1 / (1 + distance)
            
        else:
            raise ValueError(f"Unknown similarity method: {method}")
        
        return float(similarity)


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'embedding_dim': 384,
        'embedding_method': 'sentence-transformers',
        'sentence_transformer_model': 'all-MiniLM-L6-v2',
        'max_keywords': 10,
        'max_summary_length': 50,
        'custom_stopwords': ['amazon', 'product', 'item', 'purchase'],
        'use_spacy': False,
        'use_transformer_summarization': False
    }
    
    # Initialize text processor
    processor = TextProcessor(config)
    
    # Example texts
    texts = [
        "This is a test text for processing. It contains multiple sentences and should be summarized.",
        "Another text about product recommendations and user preferences."
    ]
    
    # Process texts
    results = processor.process_texts(texts)
    
    for i, result in enumerate(results):
        print(f"\nText {i+1}:")
        print(f"  Cleaned: {result.cleaned[:50]}...")
        print(f"  Summary: {result.summary[:50]}...")
        print(f"  Keywords: {result.keywords}")
        print(f"  Word count: {result.word_count}")
        print(f"  Embedding shape: {result.embedding.shape if result.embedding is not None else None}")
    
    # Compute similarity
    similarity = processor.compute_similarity(texts[0], texts[1])
    print(f"\nSimilarity between texts: {similarity:.4f}")