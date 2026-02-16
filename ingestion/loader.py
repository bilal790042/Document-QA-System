# here i need to load documents from pyPdf and others word and web scraping
# so for that i need langchain_loader
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader,TextLoader, WebBaseLoader
)
from pathlib import Path
from Models.llm import get_llm

class DocumentLoader:
    def load_file(self, file_path:str):
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            loader = PyPDFLoader(file_path)

        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)

        elif ext == '.txt':
            loader = TextLoader(file_path)

        else:
            raise ValueError('unsupported file format')
        
        return loader.load()
    

    def load_url(self, url:str):
        loader = WebBaseLoader(url)
        return loader.load()
    

