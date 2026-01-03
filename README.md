# 📚 RAG Document Assistant

A basic Retrieval-Augmented Generation (RAG) system that serves as a **standard reference structure** for building more complex RAG applications.

## 🎯 About

This project demonstrates a clean, modular RAG pipeline that can be used as a foundation for building production-grade AI document assistants. It implements the three core RAG components:

- **Data Ingestion**: PDF documents → chunking → vector embeddings
- **Data Retrieval**: Semantic search using vector similarity
- **Response Generation**: Context-aware answers using LLM

## ⚡ Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables** (`.env`):
   ```env
   MONGODB_URI=your_mongodb_connection_string
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. **Initialize vector index** (one-time):
   ```python
   from main import VectorIndexManager
   index_manager = VectorIndexManager()
   index_manager.create_or_verify_index()
   ```

4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## 🔑 Key Features

- **Modular architecture** - Easy to extend and customize
- **Vector search** - Fast semantic similarity search using MongoDB Atlas
- **Response caching** - Optimized API usage
- **Clean UI** - Streamlit interface
- **Error handling** - Robust error management and user feedback

## 💡 Use Cases

This reference implementation can be adapted for:
- Legal/medical/financial document analysis
- Document Q&A systems
- Research assistants


## 📝 Note

This is a **basic standard structure** designed for rapid prototyping. It can be improved for production.

## 📚 Resources

**MongoDB Atlas Vector Search Documentation:**  
[https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/vector-search-quick-start/?deployment-type=atlas&interface-atlas-only=driver&language-atlas-only=python](https://www.mongodb.com/docs/atlas/atlas-vector-search/tutorials/vector-search-quick-start/?deployment-type=atlas&interface-atlas-only=driver&language-atlas-only=python)


## Screenshots

![alt text](image.png)
---
