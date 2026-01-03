import streamlit as st
import os
from datetime import datetime
from main import (
    DataIngestion, 
    ResponseGeneration, 
    VectorIndexManager,
    CONFIG
)

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'index_initialized' not in st.session_state:
    st.session_state.index_initialized = False

# Header
st.markdown('<p class="main-header">📚 RAG Document Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your documents using AI-powered search</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    
    # System Status
    st.subheader("📊 System Status")
    
    # Check MongoDB connection
    try:
        mongodb_uri = os.getenv("MONGODB_URI")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if mongodb_uri and gemini_key:
            st.success("✓ Environment configured")
        else:
            st.error("✗ Missing environment variables")
    except:
        st.error("✗ Configuration error")
    
    # Display configuration
    with st.expander("🔧 Current Settings"):
        st.write(f"**Model:** {CONFIG['GENERATION_MODEL']}")
        st.write(f"**Chunk Size:** {CONFIG['CHUNK_SIZE']}")
        st.write(f"**Search Limit:** {CONFIG['SEARCH_LIMIT']}")
        st.write(f"**Embedding Dims:** {CONFIG['EMBEDDING_DIMENSIONS']}")
    
    st.markdown("---")
    
    # Document Management
    st.subheader("📁 Document Management")
    
    # Index Management
    if st.button("🔄 Initialize/Verify Index", use_container_width=True):
        with st.spinner("Setting up vector index..."):
            try:
                index_manager = VectorIndexManager()
                index_manager.create_or_verify_index()
                index_manager.close()
                st.session_state.index_initialized = True
                st.success("✓ Index ready!")
            except Exception as e:
                st.error(f"✗ Error: {str(e)}")
    
    if st.session_state.index_initialized:
        st.success("✓ Index is ready")
    
    # File Upload
    st.markdown("#### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF document to ingest into the system"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Ingest data
                    ingestion = DataIngestion(temp_path)
                    ingestion.preprocess_data()
                    ingestion.ingest_data()
                    ingestion.close()
                    
                    st.success(f"✓ Successfully ingested {uploaded_file.name}")
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    st.error(f"✗ Ingestion failed: {str(e)}")
    
    st.markdown("---")
    
    # Clear Chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8rem;'>
            <p>Powered by Gemini & MongoDB</p>
            <p>Built with Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Welcome! Ask me anything about your documents.")
        else:
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                with st.chat_message("user"):
                    st.write(chat['question'])
                
                # Assistant response
                with st.chat_message("assistant"):
                    st.write(chat['answer'])
                    st.caption(f"🕐 {chat['timestamp']}")
    
    # Query input
    st.markdown("---")
    user_query = st.text_area(
        "Ask your question:",
        placeholder="e.g., What are the key features mentioned in the document?",
        height=100,
        key="query_input"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    
    with col_btn1:
        submit_button = st.button("🔍 Ask", use_container_width=True, type="primary")
    
    with col_btn2:
        clear_input = st.button("✖️ Clear", use_container_width=True)
    
    if clear_input:
        st.rerun()
    
    # Process query
    if submit_button and user_query.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                # Generate response
                rag = ResponseGeneration(user_query)
                answer = rag.generate_response()
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_query,
                    'answer': answer,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                # Rerun to update chat
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your vector index is initialized and documents are ingested.")
    
    elif submit_button:
        st.warning("⚠️ Please enter a question first.")

with col2:
    st.subheader("📖 Quick Guide")
    
    with st.expander("🚀 Getting Started", expanded=True):
        st.markdown("""
        **Step 1:** Initialize the vector index
        - Click "Initialize/Verify Index" in sidebar
        
        **Step 2:** Upload your documents
        - Use the file uploader to add PDFs
        - Click "Ingest Document" to process
        
        **Step 3:** Ask questions
        - Type your question in the text area
        - Click "Ask" to get answers
        """)
    
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - What is this document about?
        - Summarize the key points
        - Who is the target audience?
        - What are the main features?
        - Explain the pricing structure
        """)
    
    with st.expander("⚙️ How It Works"):
        st.markdown("""
        **RAG Pipeline:**
        1. **Document Ingestion**: PDFs are split into chunks
        2. **Embedding**: Each chunk is converted to vectors
        3. **Storage**: Vectors stored in MongoDB Atlas
        4. **Retrieval**: Semantic search finds relevant chunks
        5. **Generation**: Gemini creates contextual answers
        """)
    
    # Statistics
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("📈 Session Stats")
        st.metric("Total Queries", len(st.session_state.chat_history))

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85rem; padding: 1rem;'>
        <p>⚡ Powered by <b>Gemini 2.0</b> | 🗄️ <b>MongoDB Atlas</b> | 🚀 <b>Streamlit</b></p>
        <p style='margin-top: 0.5rem;'>Need help? Check the Quick Guide on the right →</p>
    </div>
""", unsafe_allow_html=True)