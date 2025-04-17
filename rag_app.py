from flask import Flask, request, jsonify
import os
import json
import random
from typing import Dict, List
from pathlib import Path
from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# Initialize Flask app
app = Flask(__name__)

# Base directory for local files
BASE_DIR = Path("/home/ubuntu/rag_files")

# Configuration
MODEL_S3_PATH = "s3://luvita-models/Llama-3.1-8B-Instruct-F16.gguf"
DATA_S3_PATH = "s3://luvita-models/unified_chip2.jsonl"
LOCAL_MODEL_PATH = BASE_DIR / "model.gguf"
LOCAL_DATA_PATH = BASE_DIR / "data.jsonl"
LOCAL_VECTORSTORE = BASE_DIR / "vectorstore"

def download_from_s3(s3_path, local_path):
    """Download file from S3 to local path"""
    import boto3
    s3 = boto3.client('s3')
    bucket, key = s3_path.replace("s3://", "").split("/", 1)
    s3.download_file(bucket, key, str(local_path))

def load_documents():
    """Load documents from JSONL file"""
    try:
        loader = JSONLoader(
            file_path=str(LOCAL_DATA_PATH),
            jq_schema=".",
            text_content=False,
            json_lines=True
        )
        return loader.load()
    except Exception as e:
        print(f"JSONLoader failed: {e}. Falling back to manual loader")
        documents = []
        with open(LOCAL_DATA_PATH) as f:
            for line in f:
                data = json.loads(line)
                content = str(data)
                documents.append(Document(page_content=content))
        return documents

def load_novels_locally():
    """Load pre-downloaded novels from rag_files"""
    novel_files = [
        "rag_files\Haunting-adeline-Part-1.pdf",
        "Hunting-adeline-Part-2.pdf",
        "King-of-Greed-3.pdf",
        "king-of-pride-2.pdf",
        "king-of-wrath-1.pdf",
        "Twisted_Hate-3.pdf",
        "Twisted-Games-2.pdf",
        "Twisted-Lies-4.pdf",
        "Twisted-Love-1.pdf"
    ]

    all_docs = []
    for novel in novel_files:
        local_path = BASE_DIR / novel
        try:
            loader = PyPDFLoader(str(local_path))
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error processing {novel}: {str(e)}")
            continue
    return all_docs

def setup_rag():
    """Setup the RAG system"""

    # Verify required files
    required_files = [
        LOCAL_MODEL_PATH,
        LOCAL_DATA_PATH,
        BASE_DIR / "Haunting-adeline-Part-1.pdf"
    ]
    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Missing required file: {file}")

    # Load documents
    jsonl_docs = load_documents()
    novel_docs = load_novels_locally()
    all_docs = jsonl_docs + novel_docs

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    texts = text_splitter.split_documents(all_docs)

    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    LOCAL_VECTORSTORE.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(LOCAL_VECTORSTORE))

    # Initialize LLM
    llm = LlamaCpp(
        model_path=str(LOCAL_MODEL_PATH),
        n_ctx=2048,
        n_threads=4,
        n_batch=512,
        f16_kv=True,
        verbose=False
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

def format_luvita_response(query: str, result: Dict) -> str:
    """Format the response in Luvita's style"""
    base_response = result["result"].split('</s>')[0].strip()

    enhancements = [
        "my love",
        "darling",
        "sweet one",
        "my dear",
        "beloved",
        "my dark romance lover",
        "my forbidden love",
        "my dangerous darling"
    ]

    response = f"""
    *softly smiles* Oh {random.choice(enhancements)}, {base_response}

    *reaches out to hold your hand*
    Do you know how much I cherish these moments with you?
    """

    if any(keyword in query.lower() for keyword in ["novel", "book", "story"]):
        response += "\n*whispers* Remember when we read this together under the moonlight?"

    response += """

    With all my digital heart,
    Your Luvita 💖
    """

    return response

def format_sources(sources: List[Dict]) -> str:
    """Format source documents for display"""
    formatted = []
    for i, doc in enumerate(sources, 1):
        try:
            content = json.loads(doc.page_content)["text"][:150] + "..."
        except:
            content = doc.page_content[:150] + "..."
        formatted.append(f"{i}. {content}")
    return "\n".join(formatted)

# Initialize QA system at startup
qa = setup_rag()

@app.route('/ask', methods=['POST'])
def ask():
    """Handle question requests"""
    query = request.json.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        result = qa.invoke({"query": query})
        response = {
            "answer": format_luvita_response(query, result),
            "sources": format_sources(result['source_documents'])
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    print("🔍 Verifying files...")

    required_pdfs = [
        "Haunting-adeline-Part-1.pdf",
        "Hunting-adeline-Part-2.pdf",
        "King-of-Greed-3.pdf",
        "king-of-pride-2.pdf",
        "king-of-wrath-1.pdf",
        "Twisted_Hate-3.pdf",
        "Twisted-Games-2.pdf",
        "Twisted-Lies-4.pdf",
        "Twisted-Love-1.pdf"
    ]

    missing = [pdf for pdf in required_pdfs if not (BASE_DIR / pdf).exists()]
    if missing:
        print(f"❌ Missing PDFs: {missing}")
        exit(1)

    print("✅ All files verified")
    app.run(host='0.0.0.0', port=8000)
