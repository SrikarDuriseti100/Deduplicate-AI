"""
Streamlit frontend for the Article Deduplication System.
Provides a user-friendly inte        st.session_state.config = {
            'embedding_provider': 'deepseek',  # Fixed to DeepSeek only
            'deepseek_api_key': base_config.get('deepseek_api_key', ''),
            'deepseek_base_url': base_config.get('deepseek_base_url', 'https://api.deepseek.com/v1'),
            'deepseek_model': base_config.get('deepseek_model', 'deepseek-chat'),
            'similarity_threshold': base_config.get('similarity_threshold', 0.7),
            'database_path': base_config.get('database_path', 'data/articles.db')
        }uploading files, configuring settings, and viewing results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import logging
from pathlib import Path
import tempfile
import os
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.deduplication_service import DeduplicationService
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Article Deduplication System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .duplicate-item {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .unique-item {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .similarity-high {
        color: #dc3545;
        font-weight: bold;
    }
    
    .similarity-medium {
        color: #fd7e14;
        font-weight: bold;
    }
    
    .similarity-low {
        color: #28a745;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'config' not in st.session_state:
        # Load configuration from .env file or Streamlit secrets
        try:
            base_config = Config.get_config()
        except:
            base_config = {}
        
        # Check Streamlit secrets for cloud deployment
        deepseek_key = ''
        if hasattr(st, 'secrets'):
            deepseek_key = st.secrets.get('DEEPSEEK_API_KEY', base_config.get('deepseek_api_key', ''))
        else:
            deepseek_key = base_config.get('deepseek_api_key', '')
            
        st.session_state.config = {
            'embedding_provider': 'deepseek',  # Fixed to DeepSeek only
            'deepseek_api_key': deepseek_key,
            'deepseek_base_url': base_config.get('deepseek_base_url', 'https://api.deepseek.com/v1'),
            'deepseek_model': base_config.get('deepseek_model', 'deepseek-chat'),
            'similarity_threshold': base_config.get('similarity_threshold', 0.7),
            'database_path': base_config.get('database_path', 'data/articles.db')
        }
    
    if 'service' not in st.session_state:
        st.session_state.service = None
    
    # Master data is now auto-loaded by the backend
    
    if 'results' not in st.session_state:
        st.session_state.results = None

def create_service():
    """Create or update the deduplication service with current config."""
    # Merge session config with loaded config for any missing values
    config = {**Config.get_config(), **st.session_state.config}
    # Enable auto-loading of master database
    st.session_state.service = DeduplicationService(config, auto_load_master=True)
    return st.session_state.service

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Article Deduplication System</h1>
        <p>AI-Powered Similarity Detection for Content Management</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the configuration sidebar."""
    st.sidebar.title("⚙️ Configuration")
    
    # AI Provider Selection
    st.sidebar.subheader("AI Provider")
    # Fixed to DeepSeek V3.1 only
    provider = "deepseek"
    st.session_state.config['embedding_provider'] = provider
    
    st.sidebar.info("🤖 **DeepSeek V3.1**\nYour configured AI provider")
    
    # API Key Status (read-only, no input fields)
    api_key = st.session_state.config.get('deepseek_api_key', '')
    
    if api_key:
        st.sidebar.success("✅ DeepSeek API key configured")
        st.sidebar.caption("🔐 API key loaded securely from .env file")
    else:
        st.sidebar.error("❌ DeepSeek API key not found")
        st.sidebar.caption("⚠️ Check your .env file configuration")

    
    # Similarity Threshold
    st.sidebar.subheader("Detection Settings")
    threshold = st.sidebar.slider(
        "Similarity Threshold:",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.config.get('similarity_threshold', 0.7),
        step=0.05,
        help="Articles with similarity above this threshold will be marked as duplicates"
    )
    st.session_state.config['similarity_threshold'] = threshold
    
    # Display current configuration
    st.sidebar.subheader("Current Settings")
    
    # Check DeepSeek status
    api_key = st.session_state.config.get('deepseek_api_key', '')
    status = '✅ Ready' if api_key else '❌ Need API Key'
    
    st.sidebar.info(f"""
    **Provider:** DEEPSEEK V3.1  
    **Threshold:** {threshold:.0%}  
    **Status:** {status}
    """)

def render_master_status_info():
    """Render master database status information."""
    st.markdown("---")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("📚 Master Database Status")
    
    # Get master data info
    service = create_service()
    master_info = service.get_master_data_info()
    
    with col2:
        if master_info['has_data']:
            st.metric("📊 Total Articles", master_info['stats']['total_articles'])
        else:
            st.metric("📊 Total Articles", "0")
    
    with col3:
        if master_info['has_data']:
            st.metric("🏷️ Categories", len(master_info['stats']['categories']))
        else:
            st.metric("🏷️ Categories", "0")
    
    # Display status
    if master_info['has_data']:
        st.success(f"✅ Master database loaded with {master_info['stats']['total_articles']} articles")
        
        # Show sample articles in an expander
        if master_info['sample_articles']:
            with st.expander("📋 Preview Sample Articles"):
                for i, article in enumerate(master_info['sample_articles'][:3], 1):
                    st.write(f"**{i}. {article['title']}**")
                    st.caption(f"Category: {article['category']} | Author: {article['author']}")
                    if i < len(master_info['sample_articles'][:3]):
                        st.divider()
    else:
        st.error("❌ Master database not found. Please ensure 'Master_Database_Articles.xlsx' is in the project directory.")
    
    st.markdown("---")

def render_deduplication_section():
    """Render the deduplication processing section."""
    st.header("🔍 Duplicate Detection")
    
    # Check if master data is available
    service = create_service()
    master_info = service.get_master_data_info()
    
    if not master_info['has_data']:
        st.error("❌ Master database not available. Please ensure 'Master_Database_Articles.xlsx' is in the project directory and restart the application.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload New Articles")
        
        uploaded_new = st.file_uploader(
            "Upload new articles file:",
            type=['xlsx', 'xls', 'csv'],
            help="Upload articles to check against the master dataset",
            key="new_upload"
        )
        
        if uploaded_new is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_new.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_new.getbuffer())
                tmp_path = tmp_file.name
            
            # Optional sheet selection
            sheet_name = None
            if uploaded_new.name.endswith(('.xlsx', '.xls')):
                try:
                    excel_file = pd.ExcelFile(tmp_path)
                    if len(excel_file.sheet_names) > 1:
                        sheet_name = st.selectbox(
                            "Select sheet:",
                            excel_file.sheet_names,
                            key="new_sheet"
                        )
                except:
                    pass
            
            if st.button("Process Articles", type="primary"):
                service = create_service()
                
                with st.spinner("Processing articles for duplicates..."):
                    result = service.process_new_articles_file(
                        tmp_path, 
                        sheet_name, 
                        st.session_state.config['similarity_threshold']
                    )
                
                if result['success']:
                    st.success(result['message'])
                    st.session_state.results = result
                else:
                    st.error(result['message'])
                
                # Clean up temporary file
                os.unlink(tmp_path)
    
    with col2:
        if st.session_state.results:
            summary = st.session_state.results['summary']
            
            st.metric("Articles Processed", summary['total_articles_processed'])
            st.metric("Duplicates Found", summary['duplicate_articles'])
            st.metric("Duplicate Rate", f"{summary['duplicate_percentage']:.1f}%")
            
            if summary['duplicate_articles'] > 0:
                st.metric("Avg Similarity", f"{summary['average_similarity_of_duplicates']:.1f}%")

def render_results_section():
    """Render the results visualization section."""
    if not st.session_state.results:
        return
    
    st.header("📊 Results Analysis")
    
    results = st.session_state.results['results']
    summary = st.session_state.results['summary']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{summary['total_articles_processed']}</h3>
            <p>Total Articles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{summary['duplicate_articles']}</h3>
            <p>Duplicates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{summary['unique_articles']}</h3>
            <p>Unique Articles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{summary['duplicate_percentage']:.1f}%</h3>
            <p>Duplicate Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of duplicates vs unique
        fig_pie = px.pie(
            values=[summary['duplicate_articles'], summary['unique_articles']],
            names=['Duplicates', 'Unique'],
            title="Duplicate Distribution",
            color_discrete_sequence=['#ff7f0e', '#2ca02c']
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Similarity distribution histogram
        similarities = [r['max_similarity_percentage'] for r in results if r['max_similarity_percentage'] > 0]
        if similarities:
            fig_hist = px.histogram(
                x=similarities,
                nbins=20,
                title="Similarity Score Distribution",
                labels={'x': 'Similarity Percentage', 'y': 'Count'}
            )
            fig_hist.add_vline(
                x=st.session_state.config['similarity_threshold'] * 100,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Detailed results
    st.subheader("Detailed Results")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        show_filter = st.selectbox(
            "Show:",
            ["All", "Duplicates Only", "Unique Only"]
        )
    
    with filter_col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Index", "Similarity (High to Low)", "Title"]
        )
    
    # Apply filters
    filtered_results = results.copy()
    
    if show_filter == "Duplicates Only":
        filtered_results = [r for r in results if r['is_duplicate']]
    elif show_filter == "Unique Only":
        filtered_results = [r for r in results if not r['is_duplicate']]
    
    # Apply sorting
    if sort_by == "Similarity (High to Low)":
        filtered_results.sort(key=lambda x: x['max_similarity_percentage'], reverse=True)
    elif sort_by == "Title":
        filtered_results.sort(key=lambda x: x['title'])
    
    # Display results
    for result in filtered_results:
        similarity_class = "similarity-high" if result['max_similarity_percentage'] >= 90 else \
                          "similarity-medium" if result['max_similarity_percentage'] >= 70 else \
                          "similarity-low"
        
        card_class = "duplicate-item" if result['is_duplicate'] else "unique-item"
        
        article_id_display = f" (ID: {result.get('article_id', 'N/A')})" if result.get('article_id') else ""
        topic_display = f" | Topic: {result.get('topic', 'N/A')}" if result.get('topic') else ""
        
        st.markdown(f"""
        <div class="{card_class}">
            <h4>#{result['index']}: {result['title']}{article_id_display}</h4>
            <p><strong>Topic:</strong> {result.get('topic', 'N/A')}{topic_display}</p>
            <p><strong>Content:</strong> {result['content']}</p>
            <p><strong>Status:</strong> {result['status'].upper()}</p>
            <p><strong>Max Similarity:</strong> <span class="{similarity_class}">{result['max_similarity_percentage']:.1f}%</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        if result['similar_articles']:
            st.write(f"**Similar Articles Found ({len(result['similar_articles'])}):**")
            for similar in result['similar_articles'][:3]:  # Show top 3
                st.write(f"• **{similar['title']}** (Similarity: {similar['similarity_percentage']:.1f}%)")
            
            if len(result['similar_articles']) > 3:
                st.write(f"... and {len(result['similar_articles']) - 3} more")
        
        st.markdown("---")

def render_export_section():
    """Render the export functionality."""
    if not st.session_state.results:
        return
    
    st.header("📤 Export Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Download Results as Excel", type="secondary"):
            service = create_service()
            
            # Create temporary file for export
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                export_success = service.export_results(
                    st.session_state.results['results'],
                    tmp_file.name
                )
                
                if export_success:
                    # Read the file and offer download
                    with open(tmp_file.name, 'rb') as f:
                        st.download_button(
                            label="📥 Download Excel File",
                            data=f.read(),
                            file_name=f"deduplication_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    st.error("Failed to export results")
                
                # Clean up
                os.unlink(tmp_file.name)
    
    with col2:
        st.info("""
        **Export includes:**
        - All processed articles
        - Similarity percentages
        - Similar articles details
        - Processing metadata
        """)

def main():
    """Main application function."""
    initialize_session_state()
    render_header()
    render_sidebar()
    
    # Main content - focused on article processing
    tab1, tab2, tab3 = st.tabs([" Process Articles", "📊 Results", "📤 Export"])
    
    # Show master database status in main area
    render_master_status_info()
    
    with tab1:
        render_deduplication_section()
    
    with tab2:
        render_results_section()
    
    with tab3:
        render_export_section()
    
    # Footer
    st.markdown("---")
    st.markdown("**Article Deduplication System** - Powered by AI for intelligent content management")

if __name__ == "__main__":
    main()