import os
import sys

# --- EMERGENCY PATH INJECTION (Keep until new env is 100% verified) ---
EXTERNAL_LIB_PATH = r'D:\ZenithRAG\external_lib'
if os.path.exists(EXTERNAL_LIB_PATH) and EXTERNAL_LIB_PATH not in sys.path:
    sys.path.insert(0, EXTERNAL_LIB_PATH)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class ZenithVectorStore:
    def __init__(self, index_path):
        """
        ZenithRAG Vector Store: Local & Quota-Free.
        Uses all-MiniLM-L6-v2 to bypass OpenAI 429 Errors.
        """
        self.index_path = index_path
        
        # Initialize local embeddings (Runs on your CPU, costs $0)
        # 2026 Standard: using the specific model repo name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.vector_store = self._load_index()

    def _load_index(self):
        """Loads the FAISS index from the D: drive if it exists."""
        if os.path.exists(self.index_path):
            try:
                # allow_dangerous_deserialization is required for local FAISS loads
                return FAISS.load_local(
                    self.index_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Index load failed: {e}")
                return None
        return None

    def add_documents(self, documents):
        """Adds new PDF/TXT chunks to the local brain and saves to D: drive."""
        # Ensure the directory for the index exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        
        # Persist to D: drive immediately for your project demo
        self.vector_store.save_local(self.index_path)
        print(f"--- ZenithRAG: Index saved to {self.index_path} ---")