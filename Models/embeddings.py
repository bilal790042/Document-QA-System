from langchain_huggingface import HuggingFaceEndpointEmbeddings

def get_embeddings():
    hugmb = HuggingFaceEndpointEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

    return hugmb