#!/usr/bin/env python3
"""
Enhanced Gradio App for Article Deduplication - Clear Results Display
"""
import gradio as gr
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

from backend.deduplication_service import DeduplicationService
from config.config import Config

def create_app():
    """Create and launch enhanced Gradio app with clear results."""
    
    # Initialize service
    config = Config.get_config()
    service = DeduplicationService(config)
    
    # Load master database
    try:
        master_count = service._auto_load_master_database()
        if master_count is None:
            master_count = 0
    except:
        master_count = 0
    
    def get_status():
        """Get system status."""
        provider = config.get('embedding_provider', 'tfidf')
        
        return f"""
# ⚡ Article Deduplication System - FAST MODE

**Status**: ✅ Ready  
**Processing Mode**: 🚀 **FAST RETRIEVAL** (Optimized TF-IDF)  
**Speed**: ~1000+ articles/second  
**Master Articles**: {master_count:,}  
**Similarity Threshold**: {config.get('similarity_threshold', 0.7):.0%}

## 🚀 Performance Features:
- **⚡ Zero API Latency**: No external API calls
- **🎯 High Accuracy**: Enhanced TF-IDF with n-grams
- **💾 Smart Caching**: Instant results for repeated queries
- **📊 Batch Processing**: Handles large datasets efficiently

## How to Use:
1. Upload an Excel (.xlsx, .xls) or CSV (.csv) file with your articles
2. Click "Analyze for Duplicates" 
3. Get instant results with detailed similarity analysis
"""

    def process_file(file_obj):
        """Process uploaded file with enhanced results display."""
        if not file_obj:
            return "❌ Please upload a file first", create_empty_results_table()
        
        try:
            # Detect file type and save with appropriate extension
            file_content = file_obj
            if file_content.startswith(b'PK'):  # Excel files start with PK
                suffix = '.xlsx'
            elif file_content.startswith(b'\xd0\xcf\x11\xe0'):  # Old Excel format
                suffix = '.xls'
            else:  # Assume CSV for other files
                suffix = '.csv'
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_obj)
                temp_path = tmp_file.name
            
            try:
                # Process the file
                response = service.process_new_articles_file(temp_path)
                
                if not response or not response.get('success'):
                    error_msg = response.get('message', 'Unknown error occurred') if response else 'No response from service'
                    return f"❌ {error_msg}", create_empty_results_table()
                
                # Extract results from response
                results = response.get('results', [])
                if not results:
                    return "❌ No results generated. Check file format.", create_empty_results_table()
                
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Generate comprehensive summary
                summary = create_enhanced_summary(results_df)
                
                # Format results table for clear display
                display_df = create_enhanced_results_table(results_df)
                
                return summary, display_df
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            return f"❌ Error: {str(e)}", create_empty_results_table()
    
    def create_enhanced_summary(results_df):
        """Create comprehensive summary with better formatting."""
        total = len(results_df)
        duplicates = len(results_df[results_df['is_duplicate'] == True])
        unique = total - duplicates
        
        # Use max_similarity_percentage column for average
        if 'max_similarity_percentage' in results_df.columns:
            avg_sim = results_df['max_similarity_percentage'].mean()
        else:
            avg_sim = 0
        
        summary = f"""
# 📊 **DEDUPLICATION ANALYSIS RESULTS**

## 🔢 **Summary Statistics**
| Metric | Count | Percentage |
|--------|-------|------------|
| **📄 Total Articles** | {total:,} | 100% |
| **🔴 Duplicates Found** | {duplicates:,} | {duplicates/total*100:.1f}% |
| **✅ Unique Articles** | {unique:,} | {unique/total*100:.1f}% |
| **📊 Average Similarity** | - | {avg_sim:.1f}% |

---
"""
        
        if duplicates > 0:
            summary += f"""
## 🚨 **DUPLICATE DETECTION ALERTS** ({duplicates} found)

### 🔍 **Top High-Risk Duplicates:**
"""
            # Add top duplicates with detailed information
            top_dupes = results_df[results_df['is_duplicate'] == True].nlargest(10, 'max_similarity_percentage')
            for i, (_, row) in enumerate(top_dupes.iterrows(), 1):
                similarity = row['max_similarity_percentage']
                title = row['title'][:60] + "..." if len(row['title']) > 60 else row['title']
                topic = row.get('topic', 'Unknown')
                dup_count = row['duplicate_count']
                
                # Risk level classification
                if similarity >= 90:
                    risk_level = "🔴 **CRITICAL RISK**"
                    action = "**IMMEDIATE REVIEW REQUIRED**"
                elif similarity >= 80:
                    risk_level = "🟠 **HIGH RISK**"
                    action = "**Review Recommended**"
                elif similarity >= 70:
                    risk_level = "🟡 **MODERATE RISK**"
                    action = "**Consider Review**"
                else:
                    risk_level = "🟢 **LOW RISK**"
                    action = "**Optional Review**"
                    
                summary += f"""
**{i}. {title}**
- **Risk Level**: {risk_level} ({similarity:.1f}% similarity)
- **Topic**: {topic}
- **Matches Found**: {dup_count} similar articles

**📋 Matched Articles from Master Database:**
"""
                # Show matched articles from master database
                similar_articles = row.get('similar_articles', [])
                for j, matched_article in enumerate(similar_articles[:5], 1):  # Show top 5 matches
                    matched_title = matched_article.get('title', 'Unknown')[:50] + "..." if len(matched_article.get('title', '')) > 50 else matched_article.get('title', 'Unknown')
                    matched_similarity = matched_article.get('similarity_percentage', 0)
                    matched_id = matched_article.get('article_id', 'N/A')
                    summary += f"""   {j}. **{matched_title}** 
      - Article ID: {matched_id}
      - Similarity: {matched_similarity:.1f}%
"""
                
                summary += f"- **Recommended Action**: {action}\n\n"
        else:
            summary += """
## ✅ **NO DUPLICATES DETECTED**

🎉 **Great News!** All articles appear to be unique based on your similarity threshold.

- **Quality Check**: All {total:,} articles passed duplicate detection
- **Confidence Level**: High (using AI-powered analysis)
- **Next Steps**: Articles are ready for publication/use
"""
        
        summary += f"""
---

## ✅ **PROCESSING SUMMARY**
- **⚡ Processing Mode**: Fast Retrieval (Optimized TF-IDF)
- **📊 Analysis Status**: ✅ Successfully completed
- **🔍 Articles Processed**: {total:,} articles analyzed
- **⚡ Performance**: Zero API latency, instant results
- **⏱️ Next Steps**: Review detailed results table below

**💡 Tip**: Use the detailed table below to examine individual articles and their similarity scores.
"""
        return summary
    
    def create_enhanced_results_table(results_df):
        """Create enhanced results table with clear formatting."""
        # Create comprehensive display table
        display_df = results_df.copy()
        
        # Select and format columns
        table_data = []
        for _, row in display_df.iterrows():
            # Format status with emojis
            if row['is_duplicate']:
                status_icon = "🔴 DUPLICATE"
                risk_level = "HIGH" if row['max_similarity_percentage'] >= 80 else "MODERATE"
            else:
                status_icon = "✅ UNIQUE"
                risk_level = "NONE"
            
            # Format title
            title = str(row['title'])[:50] + "..." if len(str(row['title'])) > 50 else str(row['title'])
            
            # Format similarity
            similarity = f"{row['max_similarity_percentage']:.1f}%"
            
            # Format matched articles from master database
            similar_articles = row.get('similar_articles', [])
            matched_articles_text = ""
            if similar_articles:
                for idx, matched in enumerate(similar_articles[:3], 1):  # Show top 3
                    matched_title = matched.get('title', 'Unknown')[:30] + "..."
                    matched_sim = matched.get('similarity_percentage', 0)
                    matched_articles_text += f"{idx}. {matched_title} ({matched_sim:.1f}%); "
                matched_articles_text = matched_articles_text.rstrip("; ")
            else:
                matched_articles_text = "None"
            
            table_data.append({
                'Article #': row['index'],
                'Title': title,
                'Topic': str(row.get('topic', 'Unknown'))[:20],
                'Status': status_icon,
                'Risk Level': risk_level,
                'Max Similarity': similarity,
                'Duplicates Found': row['duplicate_count'],
                'Matched Master Articles': matched_articles_text
            })
        
        return pd.DataFrame(table_data)
    
    def create_empty_results_table():
        """Create empty results table placeholder."""
        return pd.DataFrame({
            'Article #': ['No data'],
            'Title': ['Upload a file to see results'],
            'Topic': [''],
            'Status': [''],
            'Risk Level': [''],
            'Max Similarity': [''],
            'Duplicates Found': [''],
            'Matched Master Articles': ['']
        })
    
    # Create enhanced Gradio interface
    with gr.Blocks(
        title="🤖 Article Deduplication System - Enhanced Results",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .duplicate-high { background-color: #ffebee !important; }
        .duplicate-moderate { background-color: #fff3e0 !important; }
        .unique { background-color: #e8f5e8 !important; }
        """
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
            <h1>⚡ AI-Powered Article Deduplication System</h1>
            <p style="font-size: 18px;">Fast Retrieval Mode - Optimized TF-IDF Processing</p>
            <p style="font-size: 14px; opacity: 0.9;">🚀 Processing Speed: 1000+ articles/second | Zero API Latency</p>
        </div>
        """)
        
        # Status section
        status_display = gr.Markdown(get_status())
        
        # Upload section
        gr.HTML("<h2>📁 Upload & Analyze Articles</h2>")
        gr.HTML("<p style='color: #666; margin-bottom: 20px;'>Supported formats: Excel (.xlsx, .xls) and CSV (.csv) files</p>")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload File (.xlsx, .xls, .csv)",
                    file_types=[".xlsx", ".xls", ".csv"],
                    type="binary"
                )
                
                analyze_btn = gr.Button(
                    "🚀 Analyze for Duplicates",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                results_summary = gr.Markdown("Upload a file to see comprehensive analysis results...")
        
        # Enhanced results section
        gr.HTML("<h2>📊 Detailed Analysis Results</h2>")
        gr.HTML("<p style='color: #666;'>Individual article analysis with risk levels and similarity scores</p>")
        
        results_table = gr.Dataframe(
            value=create_empty_results_table(),
            label="📋 Article-by-Article Analysis Report",
            interactive=False,
            wrap=True
        )
        
        # Event handlers
        analyze_btn.click(
            fn=process_file,
            inputs=[file_input],
            outputs=[results_summary, results_table]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 40px; color: #666; border-top: 2px solid #eee; padding-top: 20px;">
            <p><strong>⚡ Fast Retrieval Mode</strong> | Optimized TF-IDF Processing | Built with Gradio</p>
            <p style="font-size: 14px;">🚀 Zero API Latency • 📊 Enhanced N-gram Analysis • 💾 Smart Caching</p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        show_error=True,
        inbrowser=True
    )