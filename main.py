import os
import time
import dotenv
from google import genai
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CONFIG = {
    "EMBEDDING_MODEL": "models/text-embedding-004",
    "GENERATION_MODEL": "gemini-2.5-flash",
    "EMBEDDING_DIMENSIONS": 768,
    "CHUNK_SIZE": 400,
    "CHUNK_OVERLAP": 20,
    "DB_NAME": "sample_mflix",
    "COLLECTION_NAME": "rag_pdf_search",
    "INDEX_NAME": "vector_index",
    "SEARCH_LIMIT": 5,
    "NUM_CANDIDATES": 150
}

def get_embedding(text, input_type="document"):
    """Generate embeddings using Gemini"""
    try:
        result = gemini_client.models.embed_content(
            model=CONFIG["EMBEDDING_MODEL"],
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"Error generating embedding: {e}")
        raise

class VectorIndexManager:
    """Manages MongoDB Atlas Vector Search Index"""
    
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.collection = self.client[CONFIG["DB_NAME"]][CONFIG["COLLECTION_NAME"]]
    
    def create_or_verify_index(self):
        """Create vector search index if it doesn't exist, or verify if it's queryable"""
        index_name = CONFIG["INDEX_NAME"]
        
        # Check if index already exists
        existing_indexes = list(self.collection.list_search_indexes())
        index_exists = any(idx.get("name") == index_name for idx in existing_indexes)
        
        if not index_exists:
            print(f"Creating vector search index: {index_name}")
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "numDimensions": CONFIG["EMBEDDING_DIMENSIONS"],
                            "path": "embedding",
                            "similarity": "cosine"
                        }
                    ]
                },
                name=index_name,
                type="vectorSearch"
            )
            self.collection.create_search_index(model=search_index_model)
            print(f"Index '{index_name}' created successfully!")
        else:
            print(f"Index '{index_name}' already exists.")
        
        print("Verifying index is queryable...")
        self._wait_for_index_ready(index_name)
        print(f"Index '{index_name}' is ready for querying.")
    
    def _wait_for_index_ready(self, index_name):
        """Poll until index is queryable"""
        while True:
            indices = list(self.collection.list_search_indexes(index_name))
            if indices and indices[0].get("queryable") is True:
                break
            time.sleep(5)
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


class DataIngestion:
    """Handles PDF ingestion, chunking, and embedding storage"""
    
    def __init__(self, doc_path: str):
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        self.doc_path = doc_path
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.collection = self.client[CONFIG["DB_NAME"]][CONFIG["COLLECTION_NAME"]]
        self.docs_to_insert = []
    
    def preprocess_data(self):
        """Load PDF and split into chunks with embeddings"""
        print(f"Loading PDF: {self.doc_path}")
        loader = PyPDFLoader(self.doc_path)
        data = loader.load()
        
        print(f"Splitting into chunks (size={CONFIG['CHUNK_SIZE']}, overlap={CONFIG['CHUNK_OVERLAP']})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["CHUNK_SIZE"],
            chunk_overlap=CONFIG["CHUNK_OVERLAP"]
        )
        documents = text_splitter.split_documents(data)
        
        print(f"Generating embeddings for {len(documents)} chunks...")
        self.docs_to_insert = []
        for i, doc in enumerate(documents):
            if i % 10 == 0:
                print(f"Processing chunk {i+1}/{len(documents)}")
            self.docs_to_insert.append({
                "text": doc.page_content,
                "embedding": get_embedding(doc.page_content),
                "source": self.doc_path,
                "chunk_id": i
            })
        
        print(f"Preprocessing complete: {len(self.docs_to_insert)} documents ready")
    
    def ingest_data(self):
        """Insert documents into MongoDB"""
        try:
            if not self.docs_to_insert:
                print("No documents to insert. Run preprocess_data() first.")
                return
            
            result = self.collection.insert_many(self.docs_to_insert)
            print(f"✓ Data ingested successfully! Inserted {len(result.inserted_ids)} documents.")
        except Exception as e:
            print(f"✗ Error during ingestion: {e}")
            raise
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


class DataRetrieval:
    """Handles vector search retrieval from MongoDB"""
    
    def __init__(self, query: str):
        self.query = query
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.collection = self.client[CONFIG["DB_NAME"]][CONFIG["COLLECTION_NAME"]]
    
    def retrieve_data(self):
        """Retrieve relevant documents using vector search"""
        print(f"Searching for: '{self.query}'")
        
        query_embedding = get_embedding(self.query, input_type="query")
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": CONFIG["INDEX_NAME"],
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": CONFIG["NUM_CANDIDATES"],
                    "limit": CONFIG["SEARCH_LIMIT"]
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        
        if not results:
            print("No relevant documents found.")
            return ""
        
        print(f"Found {len(results)} relevant chunks")
        
        context = "\n\n".join([doc["text"] for doc in results])
        # print(context)
        return context
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()


class ResponseGeneration:
    """Generates responses using RAG pipeline"""
    
    def __init__(self, user_query: str):
        self.user_query = user_query
    
    def generate_response(self):
        """Generate response using retrieved context"""
        
        retriever = DataRetrieval(self.user_query)
        context_doc = retriever.retrieve_data()
        retriever.close()
        
        if not context_doc:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or ask something else."
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the question accurately and concisely.

Context:
{context_doc}

Question: {self.user_query}

Answer: Provide a clear, informative answer based on the context above. If the context doesn't contain enough information, say so."""
        
        print("Generating response with Gemini...")
        
        try:
            response = gemini_client.models.generate_content(
                model=CONFIG["GENERATION_MODEL"],
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"


# Main execution examples
if __name__ == "__main__":
    
    # Example 1: Create or verify the vector index(run once)
    print("\n" + "="*60)
    print("SETTING UP VECTOR INDEX")
    print("="*60)
    # index_manager = VectorIndexManager()
    # index_manager.create_or_verify_index()
    # index_manager.close()
    
    # Example 2: Ingest new PDF (run when adding documents)
    print("\n" + "="*60)
    print("DATA INGESTION")
    print("="*60)
    # doc_path = r"E:\GenAI_Projects\Rag_MongoDB\HDFC_MF_Index_Solutions_Factsheet-April_2025.pdf"
    # ingestion = DataIngestion(doc_path)
    # ingestion.preprocess_data()
    # ingestion.ingest_data()
    # ingestion.close()
    
    # Example 3: Query the system
    print("\n" + "="*60)
    print("QUERYING RAG SYSTEM")
    print("="*60)
    user_question = "I want to know HDFC Gold ETF is suitable for whom?"
    rag = ResponseGeneration(user_question)
    answer = rag.generate_response()
    
    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    print(answer)