"""
Configuration settings for the Article Deduplication System.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # Default configuration values
    DEFAULT_CONFIG = {
        # Database settings
        'database_path': 'data/articles.db',
        
        # AI Provider settings
        'embedding_provider': 'deepseek',  # Options: 'deepseek', 'tfidf', 'openai', 'huggingface'
        'deepseek_api_key': '',
        'deepseek_base_url': 'https://api.deepseek.com/v1',
        'deepseek_model': 'deepseek-chat',
        'openai_api_key': '',
        'openai_model': 'text-embedding-ada-002',
        'huggingface_api_key': '',
        'huggingface_model': 'sentence-transformers/all-MiniLM-L6-v2',
        
        # Similarity settings
        'similarity_threshold': 0.7,
        'max_articles_to_process': 1000,
        
        # TF-IDF settings (fallback)
        'tfidf_max_features': 300,
        'tfidf_ngram_range': (1, 2),
        
        # Caching settings
        'enable_embedding_cache': True,
        'cache_file': 'data/embeddings_cache.pkl',
        
        # Logging settings
        'log_level': 'INFO',
        'log_file': 'logs/deduplication.log',
        
        # Export settings
        'export_format': 'xlsx',
        'export_directory': 'exports',
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """
        Get configuration with environment variable overrides.
        
        Returns:
            Configuration dictionary
        """
        config = cls.DEFAULT_CONFIG.copy()
        
        # Override with environment variables if available
        env_mappings = {
            'DEEPSEEK_API_KEY': 'deepseek_api_key',
            'DEEPSEEK_BASE_URL': 'deepseek_base_url', 
            'DEEPSEEK_MODEL': 'deepseek_model',
            'OPENAI_API_KEY': 'openai_api_key',
            'HUGGINGFACE_API_KEY': 'huggingface_api_key',
            'EMBEDDING_PROVIDER': 'embedding_provider',
            'SIMILARITY_THRESHOLD': 'similarity_threshold',
            'DATABASE_PATH': 'database_path',
            'LOG_LEVEL': 'log_level',
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # Handle type conversion for numeric values
                if config_key == 'similarity_threshold':
                    try:
                        config[config_key] = float(env_value)
                    except ValueError:
                        pass
                else:
                    config[config_key] = env_value
        
        return config
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> tuple:
        """
        Validate configuration settings.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate similarity threshold
        threshold = config.get('similarity_threshold', 0.7)
        if not (0.0 <= threshold <= 1.0):
            errors.append("Similarity threshold must be between 0.0 and 1.0")
        
        # Validate embedding provider
        provider = config.get('embedding_provider', 'deepseek')
        valid_providers = ['deepseek', 'tfidf', 'openai', 'huggingface']
        if provider not in valid_providers:
            errors.append(f"Embedding provider must be one of: {valid_providers}")
        
        # Validate API keys for AI providers
        if provider == 'deepseek' and not config.get('deepseek_api_key'):
            errors.append("DeepSeek API key is required when using DeepSeek provider")
        
        if provider == 'openai' and not config.get('openai_api_key'):
            errors.append("OpenAI API key is required when using OpenAI provider")
        
        if provider == 'huggingface' and not config.get('huggingface_api_key'):
            errors.append("Hugging Face API key is required when using Hugging Face provider")
        
        # Validate paths
        database_path = config.get('database_path', '')
        if database_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(database_path), exist_ok=True)
        
        return len(errors) == 0, errors


# Example configuration profiles for different use cases
DEVELOPMENT_CONFIG = {
    **Config.DEFAULT_CONFIG,
    'embedding_provider': 'tfidf',
    'log_level': 'DEBUG',
    'enable_embedding_cache': True,
}

PRODUCTION_CONFIG = {
    **Config.DEFAULT_CONFIG,
    'embedding_provider': 'deepseek',  # Using DeepSeek as primary provider
    'log_level': 'INFO',
    'enable_embedding_cache': True,
    'max_articles_to_process': 5000,
}

TESTING_CONFIG = {
    **Config.DEFAULT_CONFIG,
    'embedding_provider': 'tfidf',
    'database_path': 'test_data/test_articles.db',
    'cache_file': 'test_data/test_cache.pkl',
    'log_level': 'WARNING',
}