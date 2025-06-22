import faiss
from sentence_transformers import SentenceTransformer

# تحميل نموذج محلي بدون أي API خارجي
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_texts(texts):
    embeddings = model.encode(texts)
    return embeddings.astype('float32')  # faiss يحتاج إلى float32

def create_vector_store(documents):
    texts = [doc.page_content for doc in documents]
    vectors = embed_texts(texts)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index, texts
