import os
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
import json

from .document_loader import load_documents
from .document_splitter import split_documents
from .vectorstore import create_vector_store, embed_texts

# تحديد مسار ملف PDF
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, 'data', 'world-cup-24-objects.pdf')

# تحميل وتقسيم المستندات وبناء قاعدة البحث
try:
    documents = load_documents(file_path)
    split_docs = split_documents(documents)
    vector_store, texts = create_vector_store(split_docs)
except Exception as e:
    documents, split_docs, vector_store, texts = [], [], None, []
    print(f"Error loading PDF or building vector store: {e}")

def home(request):
    return render(request, 'main/home.html')

@csrf_exempt
def chat(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_question = data.get('question', '')
        if not user_question or not vector_store:
            return JsonResponse({'answer': 'No answer available.'})
        query_vec = embed_texts([user_question])
        D, I = vector_store.search(query_vec, k=3)
        results = [texts[i] for i in I[0] if i < len(texts)]
        answer = " ".join(results) if results else "No answer found."
        return JsonResponse({'answer': answer})
    return render(request, 'main/chat.html')
