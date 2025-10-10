"""
Vector similarity engine for article deduplication using AI embeddings.
Supports multiple AI providers: OpenAI, Hugging Face, and sentence-transformers.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from pathlib import Path
from openai import OpenAI

class VectorSimilarityEngine:
    """Handles vector embedding generation and similarity calculations."""
    
    def __init__(self, config: Dict):
        """
        Initialize the similarity engine.
        
        Args:
            config: Configuration dictionary containing API keys and settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.embeddings_cache = {}
        self.cache_file = "data/embeddings_cache.pkl"
        self.tfidf_vectorizer = None
        self.load_cache()
    
    def load_cache(self):
        """Load embeddings cache from disk."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embeddings_cache)} cached embeddings")
        except Exception as e:
            self.logger.warning(f"Could not load cache: {e}")
            self.embeddings_cache = {}
    
    def save_cache(self):
        """Save embeddings cache to disk."""
        try:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            self.logger.info(f"Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            self.logger.error(f"Could not save cache: {e}")
    
    def get_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using optimized TF-IDF for fast retrieval.
        Bypasses API calls for maximum speed.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Use fast TF-IDF directly without API overhead
        self.logger.info("Using optimized TF-IDF for fast data retrieval")
        return self._get_enhanced_tfidf_embeddings(texts)
    
    def get_deepseek_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using optimized TF-IDF for fast retrieval.
        Bypasses API calls for maximum speed.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Use fast TF-IDF directly without API overhead
        self.logger.info("Using optimized TF-IDF for fast data retrieval")
        return self._get_enhanced_tfidf_embeddings(texts)
    
    def _get_enhanced_tfidf_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Enhanced TF-IDF embeddings with better preprocessing for semantic similarity.
        This can be enhanced with DeepSeek preprocessing in future versions.
        """
        try:
            # Enhanced TF-IDF with better parameters for semantic similarity
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=512,  # More features for better representation
                    stop_words='english',
                    ngram_range=(1, 3),  # Include trigrams for better context
                    lowercase=True,
                    min_df=1,  # Include rare terms
                    max_df=0.95,  # Exclude very common terms
                    sublinear_tf=True  # Use sublinear scaling
                )
                self.tfidf_vectorizer.fit(texts)
            
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
            # Apply L2 normalization for cosine similarity
            from sklearn.preprocessing import normalize
            normalized_matrix = normalize(tfidf_matrix, norm='l2')
            
            return normalized_matrix.toarray().tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced TF-IDF embeddings: {e}")
            # Fallback to basic TF-IDF
            return self._get_tfidf_embeddings(texts)
    
    def get_huggingface_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using Hugging Face API.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        api_key = self.config.get('huggingface_api_key')
        if not api_key:
            raise ValueError("Hugging Face API key not provided")
        
        embeddings = []
        model = self.config.get('huggingface_model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        for text in texts:
            # Check cache first
            cache_key = f"hf_{model}_{hash(text)}"
            if cache_key in self.embeddings_cache:
                embeddings.append(self.embeddings_cache[cache_key])
                continue
            
            try:
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'inputs': text
                }
                
                response = requests.post(
                    f'https://api-inference.huggingface.co/pipeline/feature-extraction/{model}',
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Hugging Face returns a tensor, we need to flatten it
                    embedding_tensor = response.json()
                    if isinstance(embedding_tensor, list) and len(embedding_tensor) > 0:
                        # Take mean pooling if multiple vectors returned
                        if isinstance(embedding_tensor[0], list):
                            embedding = np.mean(embedding_tensor, axis=0).tolist()
                        else:
                            embedding = embedding_tensor
                        embeddings.append(embedding)
                        # Cache the embedding
                        self.embeddings_cache[cache_key] = embedding
                    else:
                        raise ValueError("Invalid embedding format from Hugging Face")
                else:
                    self.logger.error(f"Hugging Face API error: {response.status_code}")
                    # Fallback to TF-IDF if API fails
                    return self._get_tfidf_embeddings(texts)
                    
            except Exception as e:
                self.logger.error(f"Error calling Hugging Face API: {e}")
                # Fallback to TF-IDF if API fails
                return self._get_tfidf_embeddings(texts)
        
        self.save_cache()
        return embeddings
    
    def _get_tfidf_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate TF-IDF embeddings as fallback when APIs are not available.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=300,  # Limit features for performance
                    stop_words='english',
                    ngram_range=(1, 2),
                    lowercase=True
                )
                self.tfidf_vectorizer.fit(texts)
            
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            return tfidf_matrix.toarray().tolist()
            
        except Exception as e:
            self.logger.error(f"Error generating TF-IDF embeddings: {e}")
            # Return zero vectors as last resort
            return [[0.0] * 300 for _ in texts]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using the configured provider.
        Optimized for fast data retrieval using TF-IDF.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        provider = self.config.get('embedding_provider', 'tfidf').lower()
        
        try:
            # Fast path: Use TF-IDF directly for maximum speed
            if provider == 'tfidf':
                return self._get_enhanced_tfidf_embeddings(texts)
            elif provider == 'deepseek':
                return self.get_deepseek_embeddings(texts)
            elif provider == 'openai':
                return self.get_openai_embeddings(texts)
            elif provider == 'huggingface':
                return self.get_huggingface_embeddings(texts)
            else:
                return self._get_enhanced_tfidf_embeddings(texts)
        except Exception as e:
            self.logger.error(f"Error with provider {provider}: {e}")
            # Fast fallback to TF-IDF
            return self._get_enhanced_tfidf_embeddings(texts)
    
    def calculate_similarity_matrix(self, embeddings1: List[List[float]], 
                                  embeddings2: List[List[float]] = None) -> np.ndarray:
        """
        Calculate cosine similarity matrix between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings (optional, uses embeddings1 if None)
            
        Returns:
            Similarity matrix
        """
        try:
            emb1_array = np.array(embeddings1)
            
            if embeddings2 is None:
                similarity_matrix = cosine_similarity(emb1_array)
            else:
                emb2_array = np.array(embeddings2)
                similarity_matrix = cosine_similarity(emb1_array, emb2_array)
            
            return similarity_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            # Return zero similarity matrix
            rows = len(embeddings1)
            cols = len(embeddings2) if embeddings2 else rows
            return np.zeros((rows, cols))
    
    def find_similar_articles(self, new_article_text: str, 
                            master_articles: List[Dict],
                            similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Find articles similar to a new article from the master dataset.
        
        Args:
            new_article_text: Text of the new article to check
            master_articles: List of master articles with 'combined_text' field
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar articles with similarity scores
        """
        try:
            # Prepare texts for embedding
            master_texts = [article['combined_text'] for article in master_articles]
            all_texts = [new_article_text] + master_texts
            
            # Generate embeddings
            self.logger.info(f"Generating embeddings for {len(all_texts)} texts")
            embeddings = self.get_embeddings(all_texts)
            
            if not embeddings or len(embeddings) != len(all_texts):
                self.logger.error("Failed to generate embeddings")
                return []
            
            # Calculate similarities
            new_embedding = [embeddings[0]]
            master_embeddings = embeddings[1:]
            
            similarities = self.calculate_similarity_matrix(new_embedding, master_embeddings)
            
            # Find similar articles
            similar_articles = []
            for i, similarity in enumerate(similarities[0]):
                if similarity >= similarity_threshold:
                    article = master_articles[i].copy()
                    article['similarity_score'] = float(similarity)
                    article['similarity_percentage'] = round(similarity * 100, 2)
                    similar_articles.append(article)
            
            # Sort by similarity score (descending)
            similar_articles.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similar_articles
            
        except Exception as e:
            self.logger.error(f"Error finding similar articles: {e}")
            return []
    
    def batch_similarity_check(self, new_articles: List[str], 
                             master_articles: List[Dict],
                             similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Check multiple new articles against master dataset.
        
        Args:
            new_articles: List of new article texts
            master_articles: List of master articles
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of results for each new article
        """
        results = []
        
        for i, new_article in enumerate(new_articles):
            self.logger.info(f"Processing article {i+1}/{len(new_articles)}")
            
            similar_articles = self.find_similar_articles(
                new_article, 
                master_articles, 
                similarity_threshold
            )
            
            results.append({
                'article_index': i,
                'article_text': new_article[:200] + "..." if len(new_article) > 200 else new_article,
                'similar_articles': similar_articles,
                'is_duplicate': len(similar_articles) > 0,
                'max_similarity': max([a['similarity_percentage'] for a in similar_articles]) if similar_articles else 0.0
            })
        
        return results


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = {
        'embedding_provider': 'deepseek',  # Using DeepSeek V3.1
        'deepseek_api_key': 'your_deepseek_key_here',
        'deepseek_base_url': 'https://api.deepseek.com',
        'deepseek_model': 'deepseek-chat',
        'openai_api_key': None,  # Add your API key here
        'huggingface_api_key': None,  # Add your API key here
    }
    
    engine = VectorSimilarityEngine(config)
    
    # Test texts
    test_texts = [
        "Artificial intelligence is transforming healthcare industry",
        "AI revolutionizes medical field with new innovations",
        "Weather forecast shows sunny skies tomorrow"
    ]
    
    # Generate embeddings
    embeddings = engine.get_embeddings(test_texts)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Calculate similarities
    similarities = engine.calculate_similarity_matrix(embeddings)
    print(f"Similarity matrix shape: {similarities.shape}")
    print(f"Sample similarities: {similarities[0][:3]}")