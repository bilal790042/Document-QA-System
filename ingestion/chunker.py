# now i have loaded and can load my document now i need to
# divide into chunks and make embedding and store it into the database
# so how do we do chunking 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingestion.loader import DocumentLoader
import os
from pathlib import Path

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 810,
    chunk_overlap = 150,
    separators=['\n\n', '\n', '.', ' ', '']
)
path = 'C:\\Users\\uoy\\Desktop\\Document QA System\\test doc.docx'
# docs = DocumentLoader.load_file(self = '',file_path=path)
loader = DocumentLoader()
docs = loader.load_file(path)
# splitted_docs = splitter.split_documents(docs)
splitted_docs = []

# generate embeddings
