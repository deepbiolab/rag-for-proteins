import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
import requests
import os
from typing import List, Dict, Any
from dataclasses import dataclass

# ---- Data Models ----
@dataclass
class OllamaConfig:
    """Configuration for Ollama connection"""
    endpoint: str = "http://127.0.0.1:11434"
    model: str = "qwen:7b"

@dataclass
class DocumentChunk:
    """Represents a chunk of processed document"""
    content: str
    metadata: Dict[str, Any]

# ---- Component 1: Data Loading ----
class SAbDabLoader:
    """Handles loading and preprocessing of antibody data from SAbDab"""
    
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def load_antibody_data(self, pdb_id: str) -> List[DocumentChunk]:
        """Load and split antibody data from SAbDab"""
        url = f"https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/structureviewer/?pdb={pdb_id}"
        loader = WebBaseLoader(url)
        docs = loader.load()
        return self.text_splitter.split_documents(docs)

# ---- Component 2: Vector Store Management ----
class VectorStoreManager:
    """Manages vector storage and retrieval operations"""
    
    def __init__(self, ollama_config: OllamaConfig):
        self.ollama_config = ollama_config
        self.embeddings = OllamaEmbeddings(
            model=ollama_config.model,
            base_url=ollama_config.endpoint
        )

    def create_vectorstore(self, documents: List[DocumentChunk], pdb_id: str) -> Chroma:
        """Create and persist vector store for documents"""
        persist_directory = f"chroma_db_{pdb_id}"
        os.makedirs(persist_directory, exist_ok=True)
        
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )

# ---- Component 3: LLM Interface ----
class OllamaInterface:
    """Handles interactions with Ollama LLM"""
    
    def __init__(self, config: OllamaConfig):
        self.config = config
        self.llm = ChatOllama(
            model=config.model,
            base_url=config.endpoint
        )

    def generate_response(self, question: str, context: str) -> str:
        """Generate response using the LLM"""
        try:
            formatted_prompt = (
                "Instruction: Answer the following question based on the context below. "
                "if the answer is not contained within the context below, say \"I don't know\". \n\n"
                f"Question: {question}\n\nContext: {context}\n\nResponse:"
            )
            response = self.llm.invoke([('human', formatted_prompt)])
            return response.content.strip()
        except Exception as e:
            return f"Error generating response: {str(e)}"

# ---- Component 4: RAG Pipeline ----
class RAGPipeline:
    """Coordinates the RAG workflow"""
    
    def __init__(self, ollama_config: OllamaConfig):
        self.loader = SAbDabLoader()
        self.vector_manager = VectorStoreManager(ollama_config)
        self.llm_interface = OllamaInterface(ollama_config)

    def process_query(self, question: str, vectorstore: Chroma) -> str:
        """Process a query through the RAG pipeline"""
        try:
            # Retrieve relevant documents
            retriever = vectorstore.as_retriever(
                search_kwargs={'k': 15}
            )
            retrieved_docs = retriever.invoke(question)
            
            # Combine retrieved documents
            context = "\n\n".join(doc.page_content fo·r doc in retrieved_docs)
            
            # Generate response
            return self.llm_interface.generate_response(question, context)
        except Exception as e:
            return f"Error in RAG pipeline: {str(e)}"

# ---- Streamlit UI ----
class StreamlitUI:
    """Handles the Streamlit user interface"""
    
    def __init__(self):
        self.config = OllamaConfig()
        self.rag_pipeline = RAGPipeline(self.config)
        self.initialize_session_state()
        self.cleanup_old_vectorstores()

    @staticmethod
    def cleanup_old_vectorstores():
        """Clean up old vector stores to prevent dimension mismatch"""
        import glob
        import shutil
        for db_path in glob.glob("chroma_db_*"):
            shutil.rmtree(db_path)

    @staticmethod
    def initialize_session_state():
        """Initialize Streamlit session state"""
        if 'vectorstore' not in st.session_state:
            st.session_state.vectorstore = None
        if 'current_pdb_id' not in st.session_state:
            st.session_state.current_pdb_id = None

    @staticmethod
    def check_ollama_connection(url: str) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{url}/api/tags")
            if response.status_code == 200:
                st.success("✅ Successfully connected to Ollama server")
                return True
            st.error(f"❌ Server responded with status code: {response.status_code}")
            return False
        except Exception as e:
            st.error(f"❌ Connection error: {str(e)}")
            return False

    def run(self):
        """Run the Streamlit interface"""
        st.title("Chat with SAbDab Antibody Data 🔬")
        st.caption("This app allows you to query antibody structural information from SAbDab using RAG")

        # Check Ollama connection
        if not self.check_ollama_connection(self.config.endpoint):
            st.error("⚠️ Cannot connect to Ollama server")
            st.info("Please make sure to:\n1. Run 'ollama serve' in a terminal\n2. Wait a few seconds for the server to start")
            st.stop()

        # Get PDB ID input
        pdb_id = st.text_input("Enter Antibody PDB ID (e.g., 7nxb)", type="default")

        if pdb_id:
            try:
                # Process new PDB ID
                if pdb_id != st.session_state.current_pdb_id:
                    with st.spinner("Loading antibody data..."):
                        documents = self.rag_pipeline.loader.load_antibody_data(pdb_id)
                    
                    with st.spinner("Creating embeddings..."):
                        st.session_state.vectorstore = self.rag_pipeline.vector_manager.create_vectorstore(
                            documents, pdb_id
                        )
                    
                    st.session_state.current_pdb_id = pdb_id
                    st.success(f"Successfully loaded antibody data for PDB ID: {pdb_id}")

                # Chat interface
                prompt = st.text_input(
                    "Ask questions about the antibody structure",
                    placeholder="What species does this antibody originate from?"
                )

                if prompt:
                    with st.spinner(f"Generating response using {self.config.model}..."):
                        result = self.rag_pipeline.process_query(
                            prompt, st.session_state.vectorstore
                        )
                        st.write(result)
                        

                # Session management
                if st.button("Clear Session"):
                    st.session_state.vectorstore = None
                    st.session_state.current_pdb_id = None
                    st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Please check:\n1. Your internet connection\n2. If the PDB ID is valid\n3. If Ollama is still running")

# ---- Main Application ----
def main():
    ui = StreamlitUI()
    ui.run()

if __name__ == "__main__":
    main()