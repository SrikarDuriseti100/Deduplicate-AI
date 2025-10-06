# Article Deduplication System

AI-powered duplicate detection for article content using vector similarity search.

## Quick Start

1. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```cmd
   cd frontend
   streamlit run app.py
   ```

3. **Use the web interface:**
   - Load your master XLSX file in "Master Data" tab
   - Upload new articles to check in "Process Articles" tab  
   - View similarity percentages and results
   - Export findings to Excel

## File Requirements

Your Excel files need:
- **Required**: `title` column
- **Optional**: `content`, `author`, `category` columns

## AI Providers

- **TF-IDF**: Free, no API key needed (default)
- **OpenAI**: Best accuracy, requires API key  
- **Hugging Face**: Good balance, requires API key

Set API keys in the sidebar when running the app.

## Results

The system shows similarity percentages for each article:
- **90%+**: Very likely duplicates
- **70-89%**: Probable duplicates  
- **50-69%**: Possible duplicates
- **<50%**: Likely unique