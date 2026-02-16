# from Models.llm import get_llm
# from retrievers.base import get_retriver
# from chains.rag_chain import build_rag_chain, format_docs

# class QAService:

#     def __init__(self):
#         self.llm = get_llm()
#         self.retriever = get_retriver(k = 4)
#         self.chain = build_rag_chain(self.llm)

#     def ask(self, question: str):
#         docs = self.retriever.invoke(question)
#         context = format_docs(docs)

#         result = self.chain.invoke({
#             'context': context, 
#             'question' : question
#         })

#         return result, docs


from Models.llm import get_llm
from retrievers.base import get_retriver
from chains.rag_chain import build_rag_chain, format_docs
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class QAService:

    def __init__(self):
        self.llm = get_llm()
        # NOW unpacks both retriever and vectorstore
        self.retriever, self.vectorstore = get_retriver(k=4)
        self.chain = build_rag_chain(self.llm)
        
        # Text splitter for uploads (same settings as your chunker)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=810,
            chunk_overlap=150,
            separators=['\n\n', '\n', '.', ' ', '']
        )

    def ask(self, question: str):
        docs = self.retriever.invoke(question)
        context = format_docs(docs)

        result = self.chain.invoke({
            'context': context, 
            'question': question
        })

        return result, docs
    
    def add_document(self, text_content: str, metadata: dict):
        """
        Add a new document to the FAISS vector store
        
        Args:
            text_content: The text content of the uploaded file
            metadata: dict with 'source' key containing filename
        
        Returns:
            Number of chunks added
        """
        # Create document
        doc = Document(page_content=text_content, metadata=metadata)
        
        # Split into chunks (using same settings as ingestion/chunker.py)
        chunks = self.text_splitter.split_documents([doc])
        
        # Add to FAISS vectorstore
        self.vectorstore.add_documents(chunks)
        
        print(f"✅ Added {len(chunks)} chunks from {metadata.get('source', 'unknown')} to FAISS")
        return len(chunks)