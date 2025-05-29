import streamlit as st
import os
import tempfile
from typing import List, Optional
import PyPDF2
import docx
from io import BytesIO
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import requests
import json
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings.base import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    """Custom SentenceTransformer embeddings wrapper"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()

class DocumentProcessor:
    """Enhanced document processor for multiple file types"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    text += page.extract_text() + "\n"
                except Exception as e:
                    st.warning(f"Error reading page {page_num + 1}: {str(e)}")
                    continue
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
            return text
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_doc(file) -> str:
        """Extract text from DOC file (basic support)"""
        try:
            # For DOC files, we'll try to read as text (limited support)
            content = file.read()
            try:
                text = content.decode('utf-8', errors='ignore')
            except:
                text = content.decode('latin-1', errors='ignore')
            return text
        except Exception as e:
            st.error(f"Error reading DOC: {str(e)}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file) -> str:
        """Extract text from TXT file"""
        try:
            content = file.read()
            if isinstance(content, bytes):
                return content.decode('utf-8', errors='ignore')
            return content
        except Exception as e:
            st.error(f"Error reading TXT: {str(e)}")
            return ""
    
    def process_uploaded_files(self, uploaded_files) -> List[Document]:
        """Process uploaded files and return list of Document objects"""
        documents = []
        
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            if file_extension == 'pdf':
                text = self.extract_text_from_pdf(uploaded_file)
                file_type = "PDF"
            elif file_extension == 'docx':
                text = self.extract_text_from_docx(uploaded_file)
                file_type = "DOCX"
            elif file_extension == 'doc':
                text = self.extract_text_from_doc(uploaded_file)
                file_type = "DOC"
            elif file_extension == 'txt':
                text = self.extract_text_from_txt(uploaded_file)
                file_type = "TXT"
            else:
                st.warning(f"Unsupported file format: {uploaded_file.name}")
                continue
            
            if text.strip():
                doc = Document(
                    page_content=text,
                    metadata={"source": uploaded_file.name, "type": file_type}
                )
                documents.append(doc)
                st.success(f"✅ {file_type} uploaded successfully: {uploaded_file.name}")
            else:
                st.warning(f"No readable text found in: {uploaded_file.name}")
        
        return documents

class FAISSVectorStore:
    """Custom FAISS vector store implementation"""
    
    def __init__(self, embeddings: SentenceTransformerEmbeddings):
        self.embeddings = embeddings
        self.index = None
        self.documents = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        # Extract text from documents
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Add embeddings to index
        self.index.add(embeddings_array)
        
        # Store documents
        self.documents.extend(documents)
        
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Get query embedding
        query_embedding = np.array([self.embeddings.embed_query(query)]).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        # Return matching documents
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results

class OllamaLLM:
    """Ollama LLM client for local Llama model"""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        
    def generate(self, prompt: str) -> str:
        """Generate response using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1024
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Please make sure Ollama is running with: 'ollama serve' and the model is available with: 'ollama pull llama3.1:8b'"
        except Exception as e:
            return f"Error: {str(e)}"

class FusionRAGChatbot:
    """Advanced Fusion RAG Chatbot"""
    
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings()
        self.vectorstore = FAISSVectorStore(self.embeddings)
        self.llm = OllamaLLM()
        self.document_processor = DocumentProcessor()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def process_documents(self, uploaded_files) -> bool:
        """Process and store documents"""
        try:
            # Process uploaded files
            documents = self.document_processor.process_uploaded_files(uploaded_files)
            
            if not documents:
                st.error("No valid documents found")
                return False
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            if not chunks:
                st.error("No text chunks created from documents")
                return False
            
            # Add to vector store
            self.vectorstore.add_documents(chunks)
            
            return True
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        try:
            # Retrieve relevant documents
            relevant_docs = self.vectorstore.similarity_search(question, k=3)
            
            if not relevant_docs:
                return "I don't have any relevant information to answer your question."
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create prompt
            prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context. 
            If you cannot find the answer in the context, say "I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate response
            response = self.llm.generate(prompt)
            
            return response
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def apply_custom_css():
    """Apply premium custom CSS styling"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #222;
        font-family: 'Poppins', sans-serif;
        color: #fff;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(34, 34, 34, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(34, 34, 34, 0.2);
        margin-top: 1rem;
    }
    
    /* Title styling */
    .main-title {
        background: linear-gradient(45deg, #999, #BBB, #DDD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #ddd;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #333;
        border-radius: 15px;
        border: 1px solid rgba(51, 51, 51, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #555;
        color: #fff;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background-color: #777;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background: rgba(51, 51, 51, 0.3);
        border-radius: 15px;
        border: 2px dashed rgba(85, 85, 85, 0.5);
        padding: 2rem;
        text-align: center;
    }
    
    /* Chat input styling */
    .stChatInput > div {
        background: rgba(51, 51, 51, 0.3);
        border-radius: 25px;
        border: 1px solid rgba(85, 85, 85, 0.2);
    }
    
    /* Success/Error message styling */
    .stSuccess {
        background: linear-gradient(45deg, #2ECC71, #27AE60);
        border-radius: 10px;
        color: white;
    }
    
    .stError {
        background: linear-gradient(45deg, #E74C3C, #C0392B);
        border-radius: 10px;
        color: white;
    }
    
    /* Logo styling */
    .logo-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .logo {
        font-size: 4rem;
        background: linear-gradient(45deg, #999, #BBB, #DDD, #EEE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 3s ease infinite;
        background-size: 400% 400%;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Card styling */
    .feature-card {
        background: rgba(51, 51, 51, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(85, 85, 85, 0.2);
        backdrop-filter: blur(5px);
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #4ECDC4 !important;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Aslami Fusion RAG Chatbot",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Header
    st.markdown('<div class="logo-container"><div class="logo">🚀</div></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Aslami Fusion RAG Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced AI-Powered Document Intelligence System</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = FusionRAGChatbot()
    
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📁 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Choose your documents",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'doc'],
            help="Upload PDF, TXT, DOC, or DOCX files"
        )
        
        if uploaded_files and st.button("🚀 Process Documents", use_container_width=True):
            with st.spinner("🔮 Harnessing the power of AI to understand your documents..."):
                success = st.session_state.chatbot.process_documents(uploaded_files)
                
                if success:
                    st.session_state.documents_processed = True
                    st.success("✨ Documents loaded successfully! Ask your questions now.")
                else:
                    st.error("Failed to process documents. Please try again.")
        
        # Model info
        st.markdown("### 🧠 Model Information")
        st.info("**LLM:** Llama 3.1-8B (Local)")
        st.info("**Embeddings:** all-MiniLM-L6-v2")
        st.info("**Vector DB:** FAISS")
        
        # Features
        st.markdown("### ✨ Features")
        st.markdown("""
        - 🔄 Local LLM Processing
        - 📚 Multi-format Support
        - 🎯 Semantic Search
        - 💾 FAISS Vector Storage
        - 🚀 Real-time Responses
        """)
    
    # Main content area
    if st.session_state.documents_processed:
        st.markdown("### 💬 Chat with Your Documents")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask your questions..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤖 Generating response..."):
                    response = st.session_state.chatbot.query(prompt)
                    
                    st.subheader("Answer")
                    st.write(response)
                    
                    # Add assistant message to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    else:
        # Welcome section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>📄 Multi-Format Support</h3>
                <p>Upload PDF, DOC, DOCX, and TXT files with intelligent text extraction.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🧠 Advanced AI</h3>
                <p>Powered by Llama 3.1-8B model running locally for privacy and speed.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 Smart Search</h3>
                <p>Semantic search with FAISS vector database for precise answers.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Getting Started")
        st.markdown("""
        1. **Upload Documents** - Use the sidebar to upload your files
        2. **Process Documents** - Click the process button to analyze your content  
        3. **Start Chatting** - Ask questions about your documents
        
        **Prerequisites:**
        - Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
        - Pull Llama model: `ollama pull llama3.1:8b`
        - Start Ollama server: `ollama serve`
        """)

if __name__ == "__main__":
    main()
