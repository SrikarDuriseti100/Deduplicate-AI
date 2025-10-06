"""
Database handler for managing article master data from Excel files.
Provides functionality to load, store, and retrieve article records.
"""

import pandas as pd
import sqlite3
import logging
from typing import List, Dict, Optional
from pathlib import Path
import hashlib

class DatabaseHandler:
    """Handles database operations for article master data."""
    
    def __init__(self, db_path: str = "data/articles.db"):
        """Initialize database handler with SQLite database."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the database and create tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    article_id TEXT,
                    title TEXT NOT NULL,
                    content TEXT,
                    author TEXT,
                    affiliation TEXT,
                    address TEXT,
                    city TEXT,
                    state TEXT,
                    zipcode TEXT,
                    email TEXT,
                    phone TEXT,
                    publication_date TEXT,
                    category TEXT,
                    tags TEXT,
                    url TEXT,
                    hash_key TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_hash_key ON articles(hash_key)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_title ON articles(title)
            """)
            
            conn.commit()
    
    def load_master_data_from_excel(self, excel_path: str, sheet_name: str = None) -> bool:
        """
        Load master data from Excel file into the database.
        
        Args:
            excel_path: Path to the Excel file
            sheet_name: Name of the sheet to read (optional)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read Excel file
            if sheet_name:
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(excel_path)
            
            # Standardize column names (case insensitive mapping)
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
                elif 'affiliation' in col_lower:
                    column_mapping[col] = 'affiliation'
                elif 'address' in col_lower:
                    column_mapping[col] = 'address'
                elif 'city' in col_lower:
                    column_mapping[col] = 'city'
                elif 'state' in col_lower:
                    column_mapping[col] = 'state'
                elif 'zipcode' in col_lower or 'zip' in col_lower:
                    column_mapping[col] = 'zipcode'
                elif 'email' in col_lower:
                    column_mapping[col] = 'email'
                elif 'phone' in col_lower:
                    column_mapping[col] = 'phone'
                elif 'date' in col_lower or 'publication_date' in col_lower:
                    column_mapping[col] = 'publication_date'
                elif 'category' in col_lower:
                    column_mapping[col] = 'category'
                elif 'tag' in col_lower:
                    column_mapping[col] = 'tags'
                elif 'url' in col_lower or 'link' in col_lower:
                    column_mapping[col] = 'url'
                elif 'article_id' in col_lower or col_lower == 'id':
                    column_mapping[col] = 'article_id'
            
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['title']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Required column '{col}' not found in Excel file")
                    return False
            
            # Fill missing columns with empty strings
            for col in ['content', 'author', 'affiliation', 'address', 'city', 'state', 'zipcode', 
                       'email', 'phone', 'publication_date', 'category', 'tags', 'url', 'article_id']:
                if col not in df.columns:
                    df[col] = ''
            
            # Generate hash keys for deduplication
            df['hash_key'] = df.apply(lambda row: self._generate_hash_key(row), axis=1)
            
            # Handle duplicate hash keys by making them unique
            df['hash_key'] = df['hash_key'] + '_' + df.index.astype(str)
            
            # Insert data into database
            with sqlite3.connect(self.db_path) as conn:
                # Clear existing data
                conn.execute("DELETE FROM articles")
                
                # Insert new data
                df.to_sql('articles', conn, if_exists='append', index=False)
                conn.commit()
            
            self.logger.info(f"Successfully loaded {len(df)} articles from {excel_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading Excel file: {str(e)}")
            return False
    
    def _generate_hash_key(self, row) -> str:
        """Generate a hash key for an article based on title and content."""
        content = f"{row.get('title', '')}{row.get('content', '')}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_all_articles(self) -> List[Dict]:
        """Retrieve all articles from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM articles ORDER BY id")
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error retrieving articles: {str(e)}")
            return []
    
    def get_article_by_id(self, article_id: int) -> Optional[Dict]:
        """Retrieve a specific article by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            self.logger.error(f"Error retrieving article {article_id}: {str(e)}")
            return None
    
    def search_articles(self, query: str, limit: int = 100) -> List[Dict]:
        """Search articles by title or content."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM articles 
                    WHERE title LIKE ? OR content LIKE ?
                    ORDER BY title
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            self.logger.error(f"Error searching articles: {str(e)}")
            return []
    
    def get_articles_for_similarity(self) -> List[Dict]:
        """Get articles formatted for similarity comparison."""
        articles = self.get_all_articles()
        return [
            {
                'id': article['id'],
                'title': article.get('title', ''),
                'content': article.get('content', ''),
                'combined_text': f"{article.get('title', '')} {article.get('content', '')}".strip(),
                'author': article.get('author', ''),
                'category': article.get('category', ''),
                'hash_key': article.get('hash_key', '')
            }
            for article in articles
        ]
    
    def get_database_stats(self) -> Dict:
        """Get statistics about the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) as total FROM articles")
                total = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT category, COUNT(*) as count 
                    FROM articles 
                    WHERE category IS NOT NULL AND category != ''
                    GROUP BY category
                    ORDER BY count DESC
                """)
                categories = [{"category": row[0], "count": row[1]} for row in cursor.fetchall()]
                
                return {
                    'total_articles': total,
                    'categories': categories
                }
        except Exception as e:
            self.logger.error(f"Error getting database stats: {str(e)}")
            return {'total_articles': 0, 'categories': []}


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    db = DatabaseHandler()
    
    # Test database initialization
    print("Database initialized successfully!")
    
    # Get stats
    stats = db.get_database_stats()
    print(f"Database stats: {stats}")