# 🤖 Article Deduplication System - Fast Mode# Article Deduplication System



AI-powered article deduplication system with optimized TF-IDF processing for instant results.AI-powered duplicate detection for article content using vector similarity search.



## ✨ Features## Quick Start



- ⚡ **Fast Retrieval Mode**: Process 1000+ articles per second### Option 1: Gradio Interface (Recommended - Modern UI)

- 🎯 **High Accuracy**: Enhanced TF-IDF with n-gram analysis

- 📊 **Detailed Results**: See matched master articles with similarity percentages1. **Install dependencies:**

- 🔍 **Risk Classification**: Automatic categorization (Critical/High/Moderate/Low)   ```cmd

- 📁 **Multiple Formats**: Supports Excel (.xlsx, .xls) and CSV files   pip install -r requirements.txt

- 💾 **Smart Caching**: Instant results for repeated queries   ```

- 📋 **Master Database**: Compare against 500 pre-loaded articles

2. **Run the Gradio application:**

## 🚀 Quick Start   ```cmd

   python gradio_app.py

### Installation   ```

   Or use the launcher:

1. Install dependencies:   ```cmd

```bash   run_gradio.bat

pip install -r requirements.txt   ```

```

3. **Access the interface:**

2. Ensure master database file exists:   - Open http://localhost:7860 in your browser

   - `Master_Database_Articles.xlsx` in root directory   - Modern, responsive UI with better visualizations

   - Easier file upload and results viewing

### Running the Application

### Option 2: Streamlit Interface (Alternative)

**Windows:**

```bash1. **Run the Streamlit application:**

run.bat   ```cmd

```   cd frontend

   streamlit run app.py

**Manual:**   ```

```bash

python app.py2. **Access at:** http://localhost:8501

```

### Usage (Both Interfaces)

The application will launch at `http://localhost:7863`- Master database auto-loads (500 articles from Excel file)

- Upload new CSV/Excel files to check for duplicates  

## 📁 Project Structure- View AI-powered similarity percentages

- Export detailed results to Excel

```

DUPLI/## File Requirements

├── app.py                          # Main Gradio application

├── backend/Your Excel files need:

│   ├── database_handler.py         # Database operations- **Required**: `title` column

│   ├── deduplication_service.py    # Main processing logic- **Optional**: `content`, `author`, `category` columns

│   └── similarity_engine.py        # TF-IDF similarity engine

├── config/## AI Providers

│   └── config.py                   # Configuration settings

├── data/- **TF-IDF**: Free, no API key needed (default)

│   ├── articles.db                 # SQLite database- **OpenAI**: Best accuracy, requires API key  

│   └── embeddings_cache.pkl        # Cached embeddings- **Hugging Face**: Good balance, requires API key

├── Master_Database_Articles.xlsx   # Master reference articles

├── .env                            # Environment configurationSet API keys in the sidebar when running the app.

├── requirements.txt                # Python dependencies

└── README.md                       # This file## Results

```

The system shows similarity percentages for each article:

## 🎯 How It Works- **90%+**: Very likely duplicates

- **70-89%**: Probable duplicates  

1. **Upload File**: Upload Excel or CSV file with articles- **50-69%**: Possible duplicates

2. **Processing**: System compares each article against 500 master articles- **<50%**: Likely unique
3. **Similarity Analysis**: Uses enhanced TF-IDF vectorization
4. **Results**: Shows duplicates with matched articles and similarity percentages

## 📊 Results Display

### Summary View
- Total articles processed
- Duplicates found (count and percentage)
- Unique articles
- Average similarity score

### Detailed Duplicate Alerts
For each duplicate:
- Article title and topic
- Risk level (Critical/High/Moderate/Low)
- **Matched articles from master database**
- **Individual similarity percentages**
- Recommended action

### Results Table
- Article number and title
- Status (Duplicate/Unique)
- Risk level
- Maximum similarity percentage
- **Matched master articles with percentages**

## ⚙️ Configuration

Edit `.env` file to customize:

```bash
# Processing Mode (optimized for speed)
EMBEDDING_PROVIDER=tfidf

# Similarity Threshold (0.7 = 70%)
SIMILARITY_THRESHOLD=0.7

# Database Path
DATABASE_PATH=data/articles.db
```

## 📋 Input File Format

Your input file should contain columns:
- `title` (required)
- `content` or `description` (recommended)
- `topic` or `category` (optional)
- `article_id` (optional)

**Example CSV:**
```csv
title,content,topic
"AI in Healthcare","Article about AI applications...",Healthcare
"Machine Learning Basics","Introduction to ML...",Technology
```

## 🎯 Performance

- **Speed**: 1000+ articles/second
- **Latency**: Zero API calls
- **Accuracy**: Enhanced TF-IDF with trigrams
- **Caching**: Smart embedding cache for repeat queries

## 🔧 Technical Details

### Similarity Engine
- **Method**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Features**: 512 dimensions
- **N-grams**: 1-3 (unigrams, bigrams, trigrams)
- **Normalization**: L2 normalization
- **Distance Metric**: Cosine similarity

### Database
- **Type**: SQLite
- **Master Articles**: 500 pre-loaded
- **Columns**: article_id, title, content, topic, category, etc.

## 📝 License

MIT License

## 🤝 Support

For issues or questions, please open an issue on GitHub.

## 🎉 Credits

Built with:
- **Gradio**: Modern UI framework
- **scikit-learn**: TF-IDF implementation
- **pandas**: Data processing
- **SQLite**: Database management

---

**Made with ❤️ for fast and accurate article deduplication**