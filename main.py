from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
import json
import time
import logging
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

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'md', 'doc'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")

if not GROQ_API_KEY or not LLAMA_PARSE_API_KEY:
    print("Warning: Missing API keys. Please set GROQ_API_KEY and LLAMA_PARSE_API_KEY in your .env file")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_data(file_path):
    try:
        parser = LlamaParse(api_key=LLAMA_PARSE_API_KEY, result_type='markdown')
        file_extractor = {
            ".docx": parser,
            ".pdf": parser,
            ".txt": parser,
            ".md": parser,
            ".doc": parser
        }
        documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()

        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        llm = Groq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

        Settings.embed_model = embed_model
        Settings.llm = llm

        chroma_client = chromadb.PersistentClient(path="./chroma_db", settings=ChromaSettings(persist_directory='./db'))
        collection_name = 'documents_collection'

        existing_collections = chroma_client.list_collections()
        collection_exists = any(col.name == collection_name for col in existing_collections)
        chroma_collection = chroma_client.get_collection(collection_name) if collection_exists else chroma_client.create_collection(name=collection_name)

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        index = VectorStoreIndex(documents, storage_context=storage_context)

        query_engine = index.as_query_engine()
        return query_engine, documents
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        raise


def summarize_document(query_engine):
    try:
        summary_prompt = PromptTemplate(
            "You are an AI assistant specializing in document analysis. "
            "Generate a concise summary capturing the main points, key insights, and essential details.\n\n"
            "Context: {context_str}\n\nSummary:"
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": summary_prompt})
        response = query_engine.query("Summarize this document comprehensively")
        return str(response)
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def process_query(query_engine, question):
    try:
        qa_prompt = PromptTemplate(
            "Strictly give answer from the context provided below.\n\n"
            "Context: {context_str}\n\nQuery: {query_str}\n\nAnswer:"
        )
        query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        return f"Error processing query: {str(e)}"


def generate_questions(documents):
    """Generate MCQ questions from document content using LLM"""
    try:
        context = "\n".join([str(doc.text) for doc in documents])

        # Initialize Groq LLM
        llm = Groq(
            model="llama3-70b-8192",
            api_key=GROQ_API_KEY
        )

        # Stricter prompt for pure JSON output
        template_string = '''
        Generate exactly 5 multiple choice questions based on the context below.

        Rules:
        - Each question must have 4 unique options
        - Include the correct answer in "answer"
        - Output must be strictly valid JSON ONLY, no extra text, no markdown, no explanations

        Context: {context}

        Required JSON format:
        [
          {{"question": "...", "options": ["...", "...", "...", "..."], "answer": "..."}},
          ...
        ]
        '''

        prompt = PromptTemplate(template=template_string)
        formatted_prompt = prompt.format(context=context)
        response = llm.complete(formatted_prompt)

        # Clean response: remove markdown/code fences or extra text
        response_text = str(response).strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        # Validate JSON
        questions = json.loads(response_text)
        if not isinstance(questions, list):
            raise ValueError("Output is not a JSON list")

        return questions

    except Exception as e:
        print("Error parsing questions:", str(e))
        # Fallback question if everything fails
        return [{
            "question": "What is the main topic of this document?",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "answer": "Option A"
        }]


def cleanup_old_files():
    try:
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(file_path) and (time.time() - os.path.getmtime(file_path)) > 3600:
                    os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up old files: {str(e)}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/app')
def app_page():
    if 'processed_file_path' not in session or not os.path.exists(session['processed_file_path']):
        return redirect(url_for('index'))
    return render_template('app.html')


@app.route("/check_document", methods=["GET"])
def check_document():
    # Assuming you store the uploaded file name in session
    filename = session.get("filename")
    if filename:
        return jsonify({"success": True, "has_document": True, "filename": filename})
    else:
        return jsonify({"success": True, "has_document": False})


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        cleanup_old_files()
        if 'file' not in request.files: return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '': return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            query_engine, documents = load_data(file_path)
            session['filename'] = filename
            session['processed_file_path'] = file_path
            return jsonify({'success': True, 'message': 'File processed successfully!', 'filename': filename})
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        return jsonify({'error': f'Upload error: {str(e)}'}), 500


def get_query_engine():
    try:
        if 'processed_file_path' not in session: return None, None
        file_path = session['processed_file_path']
        if not os.path.exists(file_path): return None, None
        return load_data(file_path)
    except Exception as e:
        return None, None


@app.route('/summarize', methods=['POST'])
def summarize():
    query_engine, _ = get_query_engine()
    if query_engine is None: return jsonify({'error': 'No document loaded'}), 400
    return jsonify({'success': True, 'summary': summarize_document(query_engine)})


@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question', '')
    if not question.strip(): return jsonify({'error': 'Question cannot be empty'}), 400
    query_engine, _ = get_query_engine()
    if query_engine is None: return jsonify({'error': 'No document loaded'}), 400
    return jsonify({'success': True, 'answer': process_query(query_engine, question)})


@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    _, documents = get_query_engine()
    if documents is None: return jsonify({'error': 'No document loaded'}), 400
    return jsonify({'success': True, 'questions': generate_questions(documents)})


@app.route('/quiz_report', methods=['POST'])
def quiz_report():
    data = request.get_json()
    session['quiz_report'] = {
        'score': data.get('score', 0),
        'total': data.get('total', 0),
        'questions': data.get('questions', [])
    }
    return jsonify({'success': True, 'report': session['quiz_report']})


@app.route('/get_quiz_report', methods=['GET'])
def get_quiz_report():
    return jsonify({'success': True, 'report': session.get('quiz_report', {})})


if __name__ == '__main__':
    app.secret_key = os.urandom(16)
    app.run(debug=True, host='0.0.0.0', port=5000)
