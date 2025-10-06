"""
Main deduplication service that orchestrates the entire deduplication process.
Combines database operations, similarity analysis, and result generation.
"""

import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import os
from datetime import datetime

from backend.database_handler import DatabaseHandler
from backend.similarity_engine import VectorSimilarityEngine
from config.config import Config

class DeduplicationService:
    """Main service for article deduplication operations."""
    
    def __init__(self, config: Dict = None, auto_load_master: bool = True):
        """
        Initialize the deduplication service.
        
        Args:
            config: Configuration dictionary (optional, will load from .env if not provided)
            auto_load_master: Whether to automatically load master database on startup
        """
        if config is None:
            self.config = Config.get_config()
        else:
            self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_handler = DatabaseHandler(self.config.get('database_path', 'data/articles.db'))
        self.similarity_engine = VectorSimilarityEngine(self.config)
        
        # Auto-load master database if enabled
        if auto_load_master:
            self._auto_load_master_database()

        
    def load_master_data(self, excel_path: str, sheet_name: str = None) -> Dict:
        """
        Load master data from Excel file.
        
        Args:
            excel_path: Path to the Excel file
            sheet_name: Sheet name (optional)
            
        Returns:
            Result dictionary with status and statistics
        """
        try:
            success = self.db_handler.load_master_data_from_excel(excel_path, sheet_name)
            
            if success:
                stats = self.db_handler.get_database_stats()
                return {
                    'success': True,
                    'message': f'Successfully loaded {stats["total_articles"]} articles',
                    'stats': stats
                }
            else:
                return {
                    'success': False,
                    'message': 'Failed to load master data',
                    'stats': {}
                }
                
        except Exception as e:
            self.logger.error(f"Error loading master data: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'stats': {}
            }
    
    def _auto_load_master_database(self):
        """Automatically load master database from the configured file."""
        try:
            # Check if master database is already loaded
            stats = self.db_handler.get_database_stats()
            if stats['total_articles'] > 0:
                self.logger.info(f"Master database already loaded with {stats['total_articles']} articles")
                return
            
            # Try to load from the default master database file
            master_file_path = "Master_Database_Articles.xlsx"
            if os.path.exists(master_file_path):
                self.logger.info(f"Auto-loading master database from {master_file_path}")
                result = self.load_master_data(master_file_path)
                if result['success']:
                    self.logger.info(f"Successfully auto-loaded master database: {result['message']}")
                else:
                    self.logger.error(f"Failed to auto-load master database: {result['message']}")
            else:
                self.logger.warning(f"Master database file not found: {master_file_path}")
                
        except Exception as e:
            self.logger.error(f"Error in auto-loading master database: {e}")
                
        except Exception as e:
            self.logger.error(f"Error loading master data: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'stats': {}
            }
    
    def process_new_articles_file(self, file_path: str, 
                                sheet_name: str = None,
                                similarity_threshold: float = 0.7) -> Dict:
        """
        Process a new file of articles and check for duplicates.
        
        Args:
            file_path: Path to the new articles file
            sheet_name: Sheet name (optional)
            similarity_threshold: Similarity threshold for duplicate detection
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Read the new articles file
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                if sheet_name:
                    new_df = pd.read_excel(file_path, sheet_name=sheet_name)
                else:
                    new_df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                new_df = pd.read_csv(file_path)
            else:
                return {
                    'success': False,
                    'message': 'Unsupported file format. Please use Excel (.xlsx, .xls) or CSV (.csv) files.',
                    'results': []
                }
            
            if new_df.empty:
                return {
                    'success': False,
                    'message': 'The uploaded file is empty.',
                    'results': []
                }
            
            # Standardize column names for new articles
            new_df = self._standardize_column_names(new_df)
            
            # Validate required columns
            if 'title' not in new_df.columns:
                return {
                    'success': False,
                    'message': 'The file must contain a "title" column.',
                    'results': []
                }
            
            # Get master articles for comparison
            master_articles = self.db_handler.get_articles_for_similarity()
            
            if not master_articles:
                return {
                    'success': False,
                    'message': 'No master data available. Please load master data first.',
                    'results': []
                }
            
            # Process each new article
            results = []
            total_articles = len(new_df)
            
            for idx, row in new_df.iterrows():
                self.logger.info(f"Processing article {idx + 1}/{total_articles}")
                
                # Prepare article text for similarity comparison
                article_id = str(row.get('article_id', f'Article_{idx+1}')).strip()
                article_title = str(row.get('title', '')).strip()
                article_content = str(row.get('content', '')).strip()
                article_topic = str(row.get('category', '')).strip()  # This is mapped from 'topic'
                # Combine title, content, and topic for better similarity matching
                combined_text = f"{article_title} {article_content} {article_topic}".strip()
                
                if not combined_text:
                    results.append({
                        'index': idx + 1,
                        'title': 'Empty Article',
                        'content': '',
                        'is_duplicate': False,
                        'similar_articles': [],
                        'max_similarity_percentage': 0.0,
                        'duplicate_count': 0,
                        'status': 'skipped',
                        'message': 'Article has no content'
                    })
                    continue
                
                # Find similar articles
                similar_articles = self.similarity_engine.find_similar_articles(
                    combined_text, 
                    master_articles, 
                    similarity_threshold
                )
                
                # Prepare result
                result = {
                    'index': idx + 1,
                    'article_id': article_id,
                    'title': article_title,
                    'topic': article_topic,
                    'content': article_content[:200] + '...' if len(article_content) > 200 else article_content,
                    'is_duplicate': len(similar_articles) > 0,
                    'similar_articles': similar_articles,
                    'max_similarity_percentage': max([a['similarity_percentage'] for a in similar_articles]) if similar_articles else 0.0,
                    'duplicate_count': len(similar_articles),
                    'status': 'duplicate' if similar_articles else 'unique',
                    'message': f'Found {len(similar_articles)} similar articles' if similar_articles else 'No duplicates found'
                }
                
                results.append(result)
            
            # Generate summary statistics
            summary = self._generate_summary(results, similarity_threshold)
            
            return {
                'success': True,
                'message': f'Successfully processed {total_articles} articles',
                'results': results,
                'summary': summary,
                'processing_timestamp': datetime.now().isoformat(),
                'similarity_threshold': similarity_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error processing new articles: {e}")
            return {
                'success': False,
                'message': f'Error processing file: {str(e)}',
                'results': []
            }
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names in the dataframe."""
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'title' in col_lower:
                column_mapping[col] = 'title'
            elif 'content' in col_lower or 'description' in col_lower or 'text' in col_lower:
                column_mapping[col] = 'content'
            elif 'topic' in col_lower or 'subject' in col_lower:
                column_mapping[col] = 'category'  # Map topic to category
            elif 'author' in col_lower or 'author_name' in col_lower:
                column_mapping[col] = 'author'
            elif 'date' in col_lower or 'publication_date' in col_lower:
                column_mapping[col] = 'publication_date'
            elif 'category' in col_lower:
                column_mapping[col] = 'category'
            elif 'article_id' in col_lower or 'id' in col_lower:
                column_mapping[col] = 'article_id'
        
        return df.rename(columns=column_mapping)
    
    def _generate_summary(self, results: List[Dict], threshold: float) -> Dict:
        """Generate summary statistics from the results."""
        total_articles = len(results)
        duplicates = [r for r in results if r['is_duplicate']]
        unique_articles = [r for r in results if not r['is_duplicate']]
        
        # Calculate average similarity for duplicates
        avg_similarity = 0.0
        if duplicates:
            similarities = [r['max_similarity_percentage'] for r in duplicates]
            avg_similarity = sum(similarities) / len(similarities)
        
        # Find articles with highest similarities
        top_similarities = sorted(
            [r for r in results if r['max_similarity_percentage'] > 0],
            key=lambda x: x['max_similarity_percentage'],
            reverse=True
        )[:5]  # Top 5 most similar
        
        return {
            'total_articles_processed': total_articles,
            'duplicate_articles': len(duplicates),
            'unique_articles': len(unique_articles),
            'duplicate_percentage': round((len(duplicates) / total_articles * 100), 2) if total_articles > 0 else 0,
            'average_similarity_of_duplicates': round(avg_similarity, 2),
            'similarity_threshold_used': threshold,
            'top_similarities': [
                {
                    'article_id': r.get('article_id', 'N/A'),
                    'title': r['title'][:50] + '...' if len(r['title']) > 50 else r['title'],
                    'topic': r.get('topic', 'N/A'),
                    'similarity_percentage': r['max_similarity_percentage'],
                    'duplicate_count': r['duplicate_count']
                }
                for r in top_similarities
            ]
        }
    
    def _auto_load_master_database(self):
        """Automatically load master database from the configured file."""
        try:
            # Check if master database is already loaded
            stats = self.db_handler.get_database_stats()
            if stats['total_articles'] > 0:
                self.logger.info(f"Master database already loaded with {stats['total_articles']} articles")
                return
            
            # Try to load from the default master database file
            master_file_path = "Master_Database_Articles.xlsx"
            if os.path.exists(master_file_path):
                self.logger.info(f"Auto-loading master database from {master_file_path}")
                result = self.load_master_data(master_file_path)
                if result['success']:
                    self.logger.info(f"Successfully auto-loaded master database: {result['message']}")
                else:
                    self.logger.error(f"Failed to auto-load master database: {result['message']}")
            else:
                self.logger.warning(f"Master database file not found: {master_file_path}")
                
        except Exception as e:
            self.logger.error(f"Error in auto-loading master database: {e}")
    
    def get_master_data_info(self) -> Dict:
        """Get information about the current master dataset."""
        try:
            stats = self.db_handler.get_database_stats()
            articles = self.db_handler.get_all_articles()
            
            # Sample articles for preview
            sample_articles = articles[:5] if articles else []
            
            return {
                'success': True,
                'stats': stats,
                'sample_articles': [
                    {
                        'id': article['id'],
                        'title': article['title'][:100] + '...' if len(article['title']) > 100 else article['title'],
                        'category': article.get('category', 'N/A'),
                        'author': article.get('author', 'N/A')
                    }
                    for article in sample_articles
                ],
                'has_data': stats['total_articles'] > 0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting master data info: {e}")
            return {
                'success': False,
                'message': f'Error: {str(e)}',
                'stats': {'total_articles': 0, 'categories': []},
                'sample_articles': [],
                'has_data': False
            }
    
    def export_results(self, results: List[Dict], output_path: str) -> bool:
        """
        Export deduplication results to Excel file.
        
        Args:
            results: Results from deduplication process
            output_path: Path for the output file
            
        Returns:
            Success status
        """
        try:
            # Prepare data for export
            export_data = []
            
            for result in results:
                base_row = {
                    'Article_Index': result['index'],
                    'Article_ID': result.get('article_id', ''),
                    'Title': result['title'],
                    'Topic': result.get('topic', ''),
                    'Content_Preview': result['content'],
                    'Status': result['status'],
                    'Is_Duplicate': result['is_duplicate'],
                    'Max_Similarity_Percentage': result['max_similarity_percentage'],
                    'Duplicate_Count': result['duplicate_count'],
                    'Message': result['message']
                }
                
                if result['similar_articles']:
                    # Create a row for each similar article
                    for i, similar in enumerate(result['similar_articles']):
                        row = base_row.copy()
                        row.update({
                            'Similar_Article_ID': similar['id'],
                            'Similar_Article_Title': similar['title'],
                            'Similar_Article_Content': similar.get('content', '')[:200],
                            'Similarity_Percentage': similar['similarity_percentage'],
                            'Similar_Article_Category': similar.get('category', ''),
                            'Similar_Article_Author': similar.get('author', '')
                        })
                        export_data.append(row)
                else:
                    # No similar articles found
                    export_data.append(base_row)
            
            # Create DataFrame and export
            df = pd.DataFrame(export_data)
            df.to_excel(output_path, index=False)
            
            self.logger.info(f"Results exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    config = {
        'database_path': 'data/articles.db',
        'embedding_provider': 'tfidf',  # Use TF-IDF for testing
        'openai_api_key': None,  # Add your API key
        'huggingface_api_key': None  # Add your API key
    }
    
    service = DeduplicationService(config)
    
    # Test master data info
    info = service.get_master_data_info()
    print(f"Master data info: {info}")