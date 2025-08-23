from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
import json
import tempfile
import shutil
import time
import logging
import re
# Document processing imports
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
import chromadb
from chromadb.config import Settings as ChromaSettings

# Load environment variables
load_dotenv() 

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'md', 'doc'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Get API keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")

if not GROQ_API_KEY or not LLAMA_PARSE_API_KEY:
    print("Warning: Missing API keys. Please set GROQ_API_KEY and LLAMA_PARSE_API_KEY in your .env file")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_data(file_path):
    """Load and process document data"""
    try:
        # Initialize parser
        parser = LlamaParse(
            api_key=LLAMA_PARSE_API_KEY,
            result_type='markdown'
        )
        # File extractor configuration
        file_extractor = {
                        ".docx": parser,
            ".pdf": parser,
            ".txt": parser,
            ".md": parser,
            ".doc": parser
        }
        # Load documents
        documents = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor=file_extractor
        ).load_data()
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        # Initialize Groq LLM
        llm = Groq(
            model="llama3-70b-8192",  # Updated to supported model
            api_key=GROQ_API_KEY
        )
        # Set global settings
        Settings.embed_model = embed_model
        Settings.llm = llm
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=ChromaSettings(persist_directory='./db')
        )
        collection_name = 'documents_collection'
        # Check if collection exists
        existing_collections = chroma_client.list_collections()
        collection_exists = any(col.name == collection_name for col in existing_collections)
        if collection_exists:
            # Get existing collection
            chroma_collection = chroma_client.get_collection(collection_name)
        else:
            # Create new collection
            chroma_collection = chroma_client.create_collection(name=collection_name)
        # Set up vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        if collection_exists:
            # Load existing index
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
        else:
            # Create new index
            index = VectorStoreIndex(documents, storage_context=storage_context)
        query_engine = index.as_query_engine()
        return query_engine, documents
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        raise

def summarize_document(query_engine):
    """Generate document summary"""
    try:
        summary_prompt = PromptTemplate(
            "You are an AI assistant specializing in document analysis. "
            "Generate a concise summary capturing the main points, key insights, and essential details. "
            "Ensure the summary retains the core meaning while being clear and informative. "
            "If the document is technical or academic, highlight its main arguments, conclusions, and any critical data points. "
            "Provide a structured summary with bullet points for better readability. "
            "Make it suitable for a student to have a quick recap.\n\n"
            "Context: {context_str}\n\n"
            "Summary:"
        )
        query_engine.update_prompts({
            "response_synthesizer:text_qa_template": summary_prompt
        })
        response = query_engine.query("Summarize this document comprehensively")
        return str(response)
    except Exception as e:
        print(f"Error in summarize_document: {str(e)}")
        return f"Error generating summary: {str(e)}"

def process_query(query_engine, question):
    """Process user query against document content"""
    try:
        qa_prompt = PromptTemplate(
            "Strictly give answer from the context provided below and follow these rules:\n"
            "- Use only the information from the provided documents to answer the query\n"
            "- Do not generate or infer information outside of the context\n"
            "- If the information is not present in the documents, reply with 'Information not found in the document'\n"
            "- Avoid any form of hallucination or assumptions\n"
            "- Provide answers that are directly supported by the context\n\n"
            "Context: {context_str}\n\n"
            "Query: {query_str}\n\n"
            "Answer:"
        )
        query_engine.update_prompts({
            "response_synthesizer:text_qa_template": qa_prompt
        })
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        return f"Error processing query: {str(e)}"

def generate_questions(documents):
    """Generate MCQ questions from document content"""
    try:
        # Extract text content from documents
        context = "\n".join([str(doc.text) for doc in documents])
        
        # Initialize Groq LLM for question generation
        llm = Groq(
            model="llama3-70b-8192",
            api_key=GROQ_API_KEY
        )
        
        template_string = '''
        Generate 5 Multiple choice type questions with 4 options and correct answer from the provided context only.
        
        Context: {context}
        
        Output format: A JSON list where each item has:
        - "question": the question text
        - "options": list of 4 different options
        - "answer": the correct answer (must be one of the options)
        
        Make sure the output is valid JSON format only without any explanation or markdown.
        '''
        
        prompt = PromptTemplate(template=template_string)
        formatted_prompt = prompt.format(context=context)
        response = llm.complete(formatted_prompt)
        
        # Convert response to string
        response_text = str(response).strip()
        
        # Remove code fences (``` or ```json)
        response_text = re.sub(r"^```(?:json)?|```$", "", response_text, flags=re.MULTILINE).strip()
        
        try:
            # Parse clean JSON
            questions = json.loads(response_text)
            return questions
        except json.JSONDecodeError as e:
            print("JSON decode failed:", e)
            # Fallback question if parsing fails
            return [{
                "question": "What is the main topic of this document?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": "Option A"
            }]
    except Exception as e:
        print(f"Error in generate_questions: {str(e)}")
        return []

def cleanup_old_files():
    """Clean up old uploaded files"""
    try:
        upload_dir = UPLOAD_FOLDER
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                # Remove files older than 1 hour
                if os.path.isfile(file_path):
                    file_age = time.time() - os.path.getmtime(file_path)
                    if file_age > 3600:  # 1 hour
                        os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up old files: {str(e)}")

@app.route('/')
def index():
    """Upload page"""
    return render_template('index.html')

@app.route('/app')
def app_page():
    """Functionalities page (Ask, Summary, Quiz)"""
    # Only allow access if a document is loaded
    if 'processed_file_path' not in session or not os.path.exists(session['processed_file_path']):
        return redirect(url_for('index'))
    return render_template('app.html')

@app.route('/debug')
def debug():
    """Debug route to inspect session state (for troubleshooting only)"""
    return {
        'session': dict(session),
        'file_exists': os.path.exists(session.get('processed_file_path', '')) if 'processed_file_path' in session else False
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        logging.info('Upload route called')
        # Clean up old files first
        cleanup_old_files()
        
        if 'file' not in request.files:
            logging.warning('No file in request.files')
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logging.warning('No file selected')
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            logging.info(f'File saved to {file_path}')
            
            # Process the file
            try:
                query_engine, documents = load_data(file_path)
                print("query_engine:---- ", query_engine)
                # Store only small metadata in session
                session['filename'] = filename
                session['processed_file_path'] = file_path
                logging.info(f'Session updated: {dict(session)}')
                # Always return JSON for AJAX/fetch
                return jsonify({
                    'success': True,
                    'message': 'File processed successfully!',
                    'filename': filename
                })
                
            except Exception as e:
                logging.error(f'Error processing file: {e}')
                # Clean up file if processing fails
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'Error processing file: {str(e)}'}), 500
        
        logging.warning('Invalid file type')
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        logging.error(f'Upload error: {e}')
        return jsonify({'error': f'Upload error: {str(e)}'}), 500

def get_query_engine():
    """Get or recreate query engine from stored data"""
    try:
        if 'processed_file_path' not in session:
            return None, None
        
        file_path = session['processed_file_path']
        if not os.path.exists(file_path):
            return None, None
        
        # Recreate the query engine
        query_engine, documents = load_data(file_path)
        return query_engine, documents
        
    except Exception as e:
        print(f"Error recreating query engine: {str(e)}")
        return None, None

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate document summary"""
    try:
        query_engine, documents = get_query_engine()
        if query_engine is None:
            return jsonify({'error': 'No document loaded or document expired'}), 400
        
        summary = summarize_document(query_engine)
        
        return jsonify({
            'success': True,
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': f'Summary error: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def query():
    """Process user query"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        query_engine, documents = get_query_engine()
        if query_engine is None:
            return jsonify({'error': 'No document loaded or document expired'}), 400
        
        answer = process_query(query_engine, question)
        
        return jsonify({
            'success': True,
            'answer': answer
        })
        
    except Exception as e:
        return jsonify({'error': f'Query error: {str(e)}'}), 500

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    """Generate quiz questions"""
    try:
        query_engine, documents = get_query_engine()
        if documents is None:
            return jsonify({'error': 'No document loaded or document expired'}), 400
        
        questions = generate_questions(documents)
        
        return jsonify({
            'success': True,
            'questions': questions
        })
        
    except Exception as e:
        return jsonify({'error': f'Quiz generation error: {str(e)}'}), 500

@app.route('/check_answer', methods=['POST'])
def check_answer():
    """Check quiz answer"""
    try:
        data = request.get_json()
        user_answer = data.get('user_answer', '')
        correct_answer = data.get('correct_answer', '')
        
        is_correct = user_answer.strip().lower() == correct_answer.strip().lower()
        
        return jsonify({
            'success': True,
            'is_correct': is_correct,
            'correct_answer': correct_answer
        })
        
    except Exception as e:
        return jsonify({'error': f'Answer check error: {str(e)}'}), 500

@app.route('/check_document', methods=['GET'])
def check_document():
    """Check if a document is currently loaded"""
    try:
        if 'processed_file_path' in session and 'filename' in session:
            file_path = session['processed_file_path']
            if os.path.exists(file_path):
                return jsonify({
                    'success': True,
                    'has_document': True,
                    'filename': session['filename']
                })
        
        return jsonify({
            'success': True,
            'has_document': False
        })
        
    except Exception as e:
        return jsonify({'error': f'Error checking document: {str(e)}'}), 500

@app.route('/clear_document', methods=['POST'])
def clear_document():
    """Clear the currently loaded document"""
    try:
        if 'processed_file_path' in session:
            file_path = session['processed_file_path']
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Clear session data
        session.pop('processed_file_path', None)
        session.pop('filename', None)
        
        return jsonify({
            'success': True,
            'message': 'Document cleared successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error clearing document: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
