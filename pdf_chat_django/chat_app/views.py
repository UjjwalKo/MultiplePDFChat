import os
import pdfplumber
import google.generativeai as genai
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extracts text from PDFs using pdfplumber for better accuracy."""
    text = ""
    for pdf in pdf_docs:
        pdf_path = os.path.join(settings.MEDIA_ROOT, pdf)
        with pdfplumber.open(pdf_path) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Ensures no NoneType errors
    return text

def get_text_chunks(text):
    """Splits text into optimized chunks for better AI processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Embeds text chunks and stores them in FAISS for retrieval."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Loads the AI model with a more detailed prompt for better responses."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Provide all relevant information, and if the answer is not in the context, say:
    "The answer is not available in the provided context."
    
    Context:\n {context}\n
    Question:\n {question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, max_tokens=1000)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Retrieves relevant text chunks and generates an AI response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    
    # Retrieve top 3 relevant chunks for better context
    docs_with_scores = new_db.similarity_search_with_score(user_question, k=3)
    docs = [doc[0] for doc in docs_with_scores]  # Extracting text only

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

def index(request):
    """Renders the main chat UI."""
    return render(request, 'chat_app/index.html')

@csrf_exempt
def process_pdfs(request):
    """Handles PDF upload and processes text into FAISS."""
    if request.method == 'POST' and request.FILES.getlist('pdf_docs'):
        pdf_docs = request.FILES.getlist('pdf_docs')
        saved_files = []
        for pdf in pdf_docs:
            file_name = default_storage.save(pdf.name, pdf)
            saved_files.append(file_name)
        
        try:
            raw_text = get_pdf_text(saved_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            return JsonResponse({'status': 'success'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'No files uploaded'})

@csrf_exempt
def chat(request):
    """Handles user questions and returns AI-generated responses."""
    if request.method == 'POST':
        user_question = request.POST.get('question')
        try:
            response = user_input(user_question)
            return JsonResponse({'status': 'success', 'response': response})
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})