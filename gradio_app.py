"""
Gradio frontend for the Article Deduplication System.
Provides a modern, user-friendly interface for uploading files and viewing results.
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import tempfile
import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.deduplication_service import DeduplicationService
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)

class GradioDeduplicationApp:
    def __init__(self):
        """Initialize the Gradio application."""
        self.config = Config.get_config()
        self.service = DeduplicationService(self.config, auto_load_master=True)
        self.results = None
        
    def get_master_info(self):
        """Get master database information."""
        master_info = self.service.get_master_data_info()
        
        if master_info['has_data']:
            stats_text = f"""
            📊 **Master Database Status**
            
            ✅ **{master_info['stats']['total_articles']} articles** loaded successfully
            🏷️ **{len(master_info['stats']['categories'])} categories** available
            
            **Sample Articles:**
            """
            
            for i, article in enumerate(master_info['sample_articles'][:3], 1):
                stats_text += f"\n{i}. **{article['title']}**\n   Category: {article['category']} | Author: {article['author']}\n"
                
            return stats_text
        else:
            return "❌ Master database not found. Please ensure 'Master_Database_Articles.xlsx' is in the project directory."
    
    def process_articles(self, file, threshold):
        """Process uploaded articles for duplicates."""
        if file is None:
            return "Please upload a file first.", None, None
        
        try:
            # Update threshold in config
            self.config['similarity_threshold'] = threshold / 100.0
            
            # Process the uploaded file
            result = self.service.process_new_articles_file(
                file.name, 
                similarity_threshold=threshold / 100.0
            )
            
            if result['success']:
                self.results = result
                
                # Generate summary text
                summary = result['summary']
                summary_text = f"""
                ## 📊 Processing Results
                
                ✅ **Successfully processed {summary['total_articles_processed']} articles**
                
                ### Summary Statistics:
                - 🔄 **Duplicates Found**: {summary['duplicate_articles']} ({summary['duplicate_percentage']:.1f}%)
                - ✨ **Unique Articles**: {summary['unique_articles']}
                - 📈 **Average Similarity**: {summary['average_similarity_of_duplicates']:.1f}% (for duplicates)
                - 🎯 **Threshold Used**: {summary['similarity_threshold_used']:.0%}
                
                ### Top Similar Articles:
                """
                
                for item in summary['top_similarities'][:3]:
                    summary_text += f"- **{item['title']}** - {item['similarity_percentage']:.1f}% similarity\n"
                
                # Create visualizations
                charts = self.create_charts(result)
                
                # Create detailed results
                detailed_results = self.format_detailed_results(result['results'])
                
                return summary_text, charts, detailed_results
                
            else:
                return f"❌ Error: {result['message']}", None, None
                
        except Exception as e:
            return f"❌ Error processing file: {str(e)}", None, None
    
    def create_charts(self, results):
        """Create visualization charts."""
        summary = results['summary']
        
        # Pie chart for duplicate distribution
        fig_pie = px.pie(
            values=[summary['duplicate_articles'], summary['unique_articles']],
            names=['Duplicates', 'Unique'],
            title="Duplicate Distribution",
            color_discrete_sequence=['#ff7f0e', '#2ca02c']
        )
        
        # Histogram for similarity distribution
        similarities = [r['max_similarity_percentage'] for r in results['results'] if r['max_similarity_percentage'] > 0]
        
        if similarities:
            fig_hist = px.histogram(
                x=similarities,
                nbins=20,
                title="Similarity Score Distribution",
                labels={'x': 'Similarity Percentage', 'y': 'Count'}
            )
            fig_hist.add_vline(
                x=summary['similarity_threshold_used'] * 100,
                line_dash="dash",
                line_color="red",
                annotation_text="Threshold"
            )
        else:
            fig_hist = px.histogram(title="No Similarity Data Available")
        
        return fig_pie, fig_hist
    
    def format_detailed_results(self, results):
        """Format detailed results for display."""
        detailed_text = "## 📋 Detailed Results\n\n"
        
        for result in results[:10]:  # Show first 10 results
            status_icon = "🔄" if result['is_duplicate'] else "✨"
            
            detailed_text += f"""
            ### {status_icon} Article #{result['index']}: {result.get('article_id', 'N/A')}
            
            **Title**: {result['title']}
            **Topic**: {result.get('topic', 'N/A')}
            **Status**: {result['status'].upper()}
            **Max Similarity**: {result['max_similarity_percentage']:.1f}%
            
            """
            
            if result['similar_articles']:
                detailed_text += f"**Similar Articles Found ({len(result['similar_articles'])}):**\n"
                for similar in result['similar_articles'][:2]:  # Show top 2 similar
                    detailed_text += f"- {similar['title']} ({similar['similarity_percentage']:.1f}% similar)\n"
                if len(result['similar_articles']) > 2:
                    detailed_text += f"- ... and {len(result['similar_articles']) - 2} more\n"
            
            detailed_text += "\n---\n\n"
            
        if len(results) > 10:
            detailed_text += f"\n*Showing first 10 results out of {len(results)} total articles.*"
            
        return detailed_text
    
    def export_results(self):
        """Export results to Excel."""
        if self.results is None:
            return None, "No results to export. Please process articles first."
        
        try:
            # Create temporary file for export
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
                export_success = self.service.export_results(
                    self.results['results'],
                    tmp_file.name
                )
                
                if export_success:
                    return tmp_file.name, "✅ Results exported successfully!"
                else:
                    return None, "❌ Failed to export results."
                    
        except Exception as e:
            return None, f"❌ Export error: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            font-family: 'Source Sans Pro', sans-serif;
        }
        .duplicate-item {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        .unique-item {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(
            theme=gr.themes.Soft(),
            css=css,
            title="Article Deduplication System"
        ) as interface:
            
            # Header
            gr.HTML("""
            <div style="text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        padding: 30px; border-radius: 15px; color: white; margin-bottom: 30px;">
                <h1 style="margin: 0; font-size: 2.5em;">🔍 Article Deduplication System</h1>
                <p style="margin: 10px 0 0 0; font-size: 1.2em;">AI-Powered Similarity Detection with DeepSeek V3.1</p>
            </div>
            """)
            
            # Configuration Status
            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #667eea;">
                        <h3>⚙️ Configuration Status</h3>
                        <p><strong>🤖 AI Provider:</strong> DeepSeek V3.1</p>
                        <p><strong>🔐 API Key:</strong> ✅ Configured securely</p>
                        <p><strong>📊 Status:</strong> Ready for processing</p>
                    </div>
                    """)
            
            # Master Database Status
            with gr.Row():
                with gr.Column():
                    master_status = gr.Markdown(
                        value=self.get_master_info(),
                        label="Master Database Information"
                    )
            
            # Main Processing Interface
            gr.HTML("<br><hr><br>")
            gr.HTML("<h2 style='text-align: center;'>🔍 Process New Articles</h2>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # File upload
                    file_upload = gr.File(
                        label="📁 Upload New Articles File",
                        file_types=[".xlsx", ".xls", ".csv"],
                        type="filepath"
                    )
                    
                    # Similarity threshold
                    threshold_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=70,
                        step=5,
                        label="🎯 Similarity Threshold (%)",
                        info="Articles above this threshold will be marked as duplicates"
                    )
                    
                    # Process button
                    process_btn = gr.Button(
                        "🚀 Process Articles",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    # Results summary
                    results_summary = gr.Markdown(
                        value="Upload a file and click 'Process Articles' to see results.",
                        label="📊 Processing Results"
                    )
            
            # Visualizations
            gr.HTML("<br><hr><br>")
            gr.HTML("<h2 style='text-align: center;'>📈 Visual Analysis</h2>")
            
            with gr.Row():
                with gr.Column():
                    pie_chart = gr.Plot(label="Duplicate Distribution")
                with gr.Column():
                    hist_chart = gr.Plot(label="Similarity Score Distribution")
            
            # Detailed Results
            gr.HTML("<br><hr><br>")
            gr.HTML("<h2 style='text-align: center;'>📋 Detailed Results</h2>")
            
            detailed_results = gr.Markdown(
                value="Process articles to see detailed similarity analysis.",
                label="Article-by-Article Analysis"
            )
            
            # Export Section
            gr.HTML("<br><hr><br>")
            gr.HTML("<h2 style='text-align: center;'>📤 Export Results</h2>")
            
            with gr.Row():
                with gr.Column():
                    export_btn = gr.Button("💾 Export to Excel", variant="secondary")
                    export_status = gr.Markdown()
                with gr.Column():
                    download_file = gr.File(label="📥 Download Results", visible=False)
            
            # Event handlers
            process_btn.click(
                fn=self.process_articles,
                inputs=[file_upload, threshold_slider],
                outputs=[results_summary, pie_chart, hist_chart, detailed_results]
            )
            
            export_btn.click(
                fn=self.export_results,
                outputs=[download_file, export_status]
            ).then(
                lambda x: gr.update(visible=True) if x[0] else gr.update(visible=False),
                inputs=[download_file],
                outputs=[download_file]
            )
            
            # Footer
            gr.HTML("""
            <div style="text-align: center; margin-top: 50px; padding: 20px; color: #666;">
                <p><strong>Article Deduplication System</strong> - Powered by DeepSeek V3.1</p>
                <p>🔐 Secure • 🚀 Fast • 🎯 Accurate</p>
            </div>
            """)
        
        return interface

def create_app():
    """Create and return the Gradio application."""
    app = GradioDeduplicationApp()
    return app.create_interface()

if __name__ == "__main__":
    # Create and launch the app
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )