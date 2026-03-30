import os
import sys
import certifi
import logging
import tempfile
import gdown
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

# --- 1. BOOTSTRAP & SYSTEM STABILITY FIXES ---
# Fix for SSL_CERT_FILE (Required for HuggingFace downloads on restricted networks)
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Emergency Path Injection (Ensures ZenithRAG finds libraries on D: drive if C: is full)
EXTERNAL_LIB_PATH = r'D:\ZenithRAG\external_lib'
if os.path.exists(EXTERNAL_LIB_PATH):
    sys.path.append(EXTERNAL_LIB_PATH)

# Initialize Flask and Logger early
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ZenithRAG")

# --- 2. SECURE ENVIRONMENT LOADER ---
def bootstrap_environment():
    """Downloads secrets from GDrive and loads .env before services start."""
    env_path = '.env'
    file_id = '1_LOl2r_idaBKdrkbvGLrChxcy8GoEonO'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    try:
        if not os.path.exists(env_path):
            logger.info("--- ZenithRAG: Fetching Cloud Secrets ---")
            gdown.download(url, env_path, quiet=False)
        load_dotenv(env_path, override=True)
        logger.info("--- ZenithRAG: Environment Ready ---")
    except Exception as e:
        logger.error(f"CRITICAL BOOTSTRAP ERROR: {e}")

# Run bootstrap before importing Config/Services
bootstrap_environment()

# --- 3. PROJECT-SPECIFIC IMPORTS (Must follow .env loading) ---
from config import Config
from models.vector_store import ZenithVectorStore as VectorStore
from services.storage_service import S3Storage
from services.llm_service import LLMService
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Safe Splitter Import (Handles version changes in 2026)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- 4. ARCHITECTURE INITIALIZATION (REINFORCED) ---
# Resolve the Vector DB Path (Hardcoded fallback to D: drive to fix [WinError 3])
raw_db_path = getattr(Config, 'VECTOR_DB_PATH', None)

if not raw_db_path or not os.path.dirname(raw_db_path):
    # If the .env path is empty or just a filename, force a valid D: folder
    vector_db_path = r'D:\ZenithRAG\faiss_index\index.faiss'
    logger.info(f"Using Default Secure Path: {vector_db_path}")
else:
    vector_db_path = raw_db_path

# Ensure the directory exists before initializing FAISS
db_dir = os.path.dirname(vector_db_path)
if db_dir:
    os.makedirs(db_dir, exist_ok=True)
    logger.info(f"Database directory verified: {db_dir}")

# Initialize Level-3 Core Services
try:
    vector_store = VectorStore(vector_db_path)
    storage_service = S3Storage()
    llm_service = LLMService(vector_store)
    logger.info("--- ZenithRAG: Architecture Initialized Successfully ---")
except Exception as e:
    logger.error(f"SERVICE INITIALIZATION FAILED: {e}")

# --- 5. CORE LOGIC ---
def process_document(file):
    """Handles PDF/TXT ingestion and recursive chunking on D: drive."""
    # Relocate temp work to D: drive to save C: drive space
    d_temp_dir = r'D:\ZenithRAG\temp'
    if not os.path.exists(d_temp_dir):
        os.makedirs(d_temp_dir)
        
    temp_dir = tempfile.mkdtemp(dir=d_temp_dir)
    temp_path = os.path.join(temp_dir, file.filename)
    try:
        file.save(temp_path)
        logger.info(f"Ingesting: {file.filename}")
        
        loader = PyPDFLoader(temp_path) if file.filename.lower().endswith('.pdf') else TextLoader(temp_path)
        documents = loader.load()
        
        # Recursive splitting provides better context for the RAG brain
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)
    finally:
        # Cleanup temp directory immediately
        if os.path.exists(temp_path): os.remove(temp_path)
        if os.path.exists(temp_dir): os.rmdir(temp_dir)

# --- 6. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    """Syncs file to Stockholm S3 and indexes locally in FAISS."""
    file = request.files.get('file')
    if not file: 
        return jsonify({'status': 'error', 'message': 'No file detected'}), 400
    try:
        text_chunks = process_document(file)
        
        # 1. Sync original file to S3 (Stockholm Region)
        file.seek(0) 
        storage_service.upload_file(file, file.filename)
        
        # 2. Add text chunks to the Local Vector Database (FAISS)
        vector_store.add_documents(text_chunks)
        
        return jsonify({'status': 'success', 'message': f'{file.filename} synced and indexed.'})
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    """Handles Groq-powered Llama 3.1 RAG query."""
    data = request.get_json(silent=True)
    if not data or not data.get('question'):
        return jsonify({'status': 'error', 'message': 'Empty query.'}), 400
    
    try:
        response = llm_service.get_response(str(data.get('question')))
        return jsonify({'status': 'success', 'response': response})
    except Exception as e:
        logger.error(f"Query error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear():
    """Resets conversational memory."""
    llm_service.clear_history()
    return jsonify({'status': 'success', 'message': 'Memory reset.'})

# --- 7. MAIN EXECUTION ---
if __name__ == '__main__':
    # ZenithRAG running on Port 8080 (Ready for Demo)
    app.run(host='0.0.0.0', port=8080, debug=True)