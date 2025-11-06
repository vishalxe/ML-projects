"""
Live Email Categorizer Web Application
Real-time interface for email categorization with Gmail integration
"""

import streamlit as st
import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
from email_service import EmailCategorizer
import plotly.express as px
import plotly.graph_objects as go
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import threading

# FastAPI app for API endpoints
api_app = FastAPI(title="Email Categorizer API")

# Add CORS middleware
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CategorizeRequest(BaseModel):
    text: str

class CategorizeResponse(BaseModel):
    category: str
    confidence: float

# API endpoints
@api_app.post("/api/categorize", response_model=CategorizeResponse)
async def categorize_email(request: CategorizeRequest):
    try:
        categorizer = get_categorizer()
        categorizer.load_model()
        category, confidence = categorizer.predict_category(request.text)
        return CategorizeResponse(category=category, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_app.get("/api/status")
async def get_status():
    return {"status": "online", "timestamp": datetime.now().isoformat()}

@api_app.get("/api/stats")
async def get_stats():
    return load_stats()

# Start API server in background
def start_api_server():
    uvicorn.run(api_app, host="0.0.0.0", port=8000, log_level="info")

# Page configuration
st.set_page_config(
    page_title="📧 Live Email Categorizer",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .category-card {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-radius: 5px;
    }
    .status-online {
        color: #28a745;
        font-weight: bold;
    }
    .status-offline {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_stats():
    """Load service statistics"""
    if os.path.exists('service_stats.json'):
        with open('service_stats.json', 'r') as f:
            return json.load(f)
    return {
        'total_processed': 0,
        'last_run': None,
        'categories': {},
        'errors': 0
    }

@st.cache_resource
def get_categorizer():
    """Get email categorizer instance"""
    return EmailCategorizer()

def main():
    # Start API server in background if not already running
    try:
        import requests
        requests.get("http://localhost:8000/api/status", timeout=1)
    except:
        # API server not running, start it
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        time.sleep(2)  # Give it a moment to start
    
    # Header
    st.markdown('<h1 class="main-header">📧 Live Email Categorizer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Control Panel")
        
        # Service status
        stats = load_stats()
        if stats['last_run']:
            last_run = datetime.fromisoformat(stats['last_run'])
            if datetime.now() - last_run < timedelta(minutes=10):
                st.markdown('<p class="status-online">🟢 Service Online</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-offline">🔴 Service Offline</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-offline">🔴 Service Not Started</p>', unsafe_allow_html=True)
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        if st.button("🚀 Start Service", type="primary"):
            st.info("To start the background service, run: `python background_service.py`")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📊 Full Categorization"):
            with st.spinner("Running full categorization..."):
                try:
                    categorizer = get_categorizer()
                    categorizer.authenticate_gmail()
                    categorizer.load_model()
                    results = categorizer.auto_categorize_emails(100)
                    st.success(f"✅ Categorized {results['processed']} emails!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📧 Live Categorization", "📋 Categorized Emails", "⚙️ Settings"])
    
    with tab1:
        st.markdown("## 📊 Service Dashboard")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Processed",
                value=stats['total_processed'],
                delta=None
            )
        
        with col2:
            st.metric(
                label="Categories Found",
                value=len(stats['categories']),
                delta=None
            )
        
        with col3:
            st.metric(
                label="Errors",
                value=stats['errors'],
                delta=None
            )
        
        with col4:
            if stats['last_run']:
                last_run = datetime.fromisoformat(stats['last_run'])
                st.metric(
                    label="Last Run",
                    value=last_run.strftime("%H:%M:%S"),
                    delta=None
                )
            else:
                st.metric(label="Last Run", value="Never", delta=None)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if stats['categories']:
                # Category distribution pie chart
                fig = px.pie(
                    values=list(stats['categories'].values()),
                    names=list(stats['categories'].keys()),
                    title="Email Categories Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categories data available yet.")
        
        with col2:
            # Processing timeline (simulated)
            if stats['last_run']:
                timeline_data = {
                    'Time': [datetime.now() - timedelta(hours=2), datetime.now() - timedelta(hours=1), datetime.now()],
                    'Emails Processed': [stats['total_processed'] // 3, stats['total_processed'] // 2, stats['total_processed']]
                }
                df_timeline = pd.DataFrame(timeline_data)
                fig = px.line(df_timeline, x='Time', y='Emails Processed', title="Processing Timeline")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No processing data available yet.")
    
    with tab2:
        st.markdown("## 📧 Live Email Categorization")
        
        # Manual categorization
        st.markdown("### Manual Categorization")
        email_text = st.text_area(
            "Enter email content:",
            height=150,
            placeholder="Paste your email content here..."
        )
        
        if st.button("🚀 Categorize Email", type="primary"):
            if email_text.strip():
                try:
                    categorizer = get_categorizer()
                    categorizer.load_model()
                    
                    category, confidence = categorizer.predict_category(email_text)
                    
                    # Display result
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Category:** {categorizer.categories[category]}")
                    with col2:
                        st.info(f"**Confidence:** {confidence:.2%}")
                    
                    # Show details
                    with st.expander("📋 Details"):
                        st.write(f"**Raw Category:** {category}")
                        st.write(f"**Email Text:** {email_text[:200]}...")
                        
                except Exception as e:
                    st.error(f"❌ Error categorizing email: {e}")
            else:
                st.warning("Please enter email content to categorize.")
        
        # File upload
        st.markdown("### Upload Email File")
        uploaded_file = st.file_uploader("Upload .txt or .pdf file", type=['txt', 'pdf'])
        
        if uploaded_file:
            try:
                if uploaded_file.type == "text/plain":
                    content = str(uploaded_file.read(), "utf-8")
                elif uploaded_file.type == "application/pdf":
                    from PyPDF2 import PdfReader
                    pdf = PdfReader(uploaded_file)
                    content = ""
                    for page in pdf.pages:
                        content += page.extract_text()
                else:
                    st.error("Unsupported file type")
                    content = ""
                
                if content:
                    st.text_area("File Content:", content, height=150)
                    
                    if st.button("📧 Categorize Uploaded File"):
                        try:
                            categorizer = get_categorizer()
                            categorizer.load_model()
                            
                            category, confidence = categorizer.predict_category(content)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.success(f"**Category:** {categorizer.categories[category]}")
                            with col2:
                                st.info(f"**Confidence:** {confidence:.2%}")
                                
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
    
    with tab3:
        st.markdown("## 📋 Categorized Emails")
        
        # Category filter
        categorizer = get_categorizer()
        category_options = ['All'] + list(categorizer.categories.keys())
        selected_category = st.selectbox("Filter by Category:", category_options)
        
        if st.button("🔄 Load Categorized Emails"):
            try:
                categorizer.authenticate_gmail()
                
                if selected_category == 'All':
                    emails = categorizer.get_categorized_emails()
                else:
                    emails = categorizer.get_categorized_emails(selected_category)
                
                if emails:
                    st.success(f"Found {len(emails)} categorized emails")
                    
                    # Display emails
                    for i, email in enumerate(emails[:20]):  # Show first 20
                        with st.expander(f"📧 {email['subject'][:50]}... ({email['sender']})"):
                            st.write(f"**From:** {email['sender']}")
                            st.write(f"**Subject:** {email['subject']}")
                            st.write(f"**Date:** {email['date']}")
                            st.write(f"**Labels:** {', '.join(email.get('labels', []))}")
                            st.write(f"**Preview:** {email['body'][:300]}...")
                else:
                    st.info("No categorized emails found. Try running the categorization process first.")
                    
            except Exception as e:
                st.error(f"❌ Error loading emails: {e}")
                st.info("Make sure you have Gmail API credentials set up.")
    
    with tab4:
        st.markdown("## ⚙️ Settings & Setup")
        
        st.markdown("### 🔑 Gmail API Setup")
        st.markdown("""
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing one
        3. Enable Gmail API
        4. Create credentials (OAuth 2.0 Client ID)
        5. Download credentials as `credentials.json`
        6. Place it in the project directory
        """)
        
        # Check if credentials exist
        if os.path.exists('credentials.json'):
            st.success("✅ Gmail credentials found")
        else:
            st.warning("⚠️ Gmail credentials not found. Please add credentials.json")
        
        st.markdown("### 🚀 Starting the Background Service")
        st.code("""
# Install dependencies
pip install -r requirements_live.txt

# Start the background service
python background_service.py
        """, language="bash")
        
        st.markdown("### 📊 Service Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Processing Schedule:**")
            st.write("• New emails: Every 5 minutes")
            st.write("• Full categorization: Every 1 hour")
            st.write("• Confidence threshold: 60%")
        
        with col2:
            st.markdown("**Categories:**")
            for key, value in categorizer.categories.items():
                st.write(f"• {value}")
        
        # Model info
        st.markdown("### 🤖 Model Information")
        if os.path.exists('email_model.pkl'):
            st.success("✅ ML model found")
        else:
            st.info("ℹ️ Model will be trained on first run")
        
        # Stats export
        if st.button("📥 Export Statistics"):
            stats_json = json.dumps(stats, indent=2)
            st.download_button(
                label="Download stats.json",
                data=stats_json,
                file_name=f"email_categorizer_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
