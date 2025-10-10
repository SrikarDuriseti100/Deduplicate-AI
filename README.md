# Article Deduplication System

AI-powered duplicate detection for article content using vector similarity search.

## Quick Start

### Option 1: Gradio Interface (Recommended - Modern UI)

1. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```

2. **Run the Gradio application:**
   ```cmd
   python gradio_app.py
   ```
   Or use the launcher:
   ```cmd
   run_gradio.bat
   ```

3. **Access the interface:**
   - Open http://localhost:7860 in your browser
   - Modern, responsive UI with better visualizations
   - Easier file upload and results viewing

### Option 2: Streamlit Interface (Alternative)

1. **Run the Streamlit application:**
   ```cmd
   cd frontend
   streamlit run app.py
   ```

2. **Access at:** http://localhost:8501

### Usage (Both Interfaces)
- Master database auto-loads (500 articles from Excel file)
- Upload new CSV/Excel files to check for duplicates  
- View AI-powered similarity percentages
- Export detailed results to Excel

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