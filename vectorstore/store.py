# so before storing it into faiss we need embeddings
from Models.embeddings import get_embeddings
from langchain_community.vectorstores import FAISS
from ingestion.chunker import splitted_docs 
# from langchain_community.docstore.in_memory import InMemoryDocstore
from Models.embeddings import get_embeddings
from langchain_core.documents import Document

# docs to be embedded
docs = splitted_docs
# text = [doc.page_content for doc in docs]
# embddings = get_embeddings().embed_documents(text)
embddings = get_embeddings()
# faiss

# vector_store = FAISS.from_documents(docs, embddings
#       )


# Start completely empty
dummy_doc = Document(page_content="Placeholder", metadata={"source": "system"})
vector_store = FAISS.from_documents([dummy_doc], embddings)

print("✅ Vectorstore ready for uploads")