# now we have models llm for invoke and llm for embeddings. moreover we have document ingestion in the form of chunker and loader we then embed them and store them in the vectorstore and after that we have a retriever for retrieved docs and query. we now need to chain somethings to see some results. for chain i will use prompts

from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from Models.llm import get_llm
from langchain_core.output_parsers import StrOutputParser
from retrievers.base import get_retriver


def build_rag_chain(llm):
    system_prompt = Path("prompts/system.txt").read_text()
    qa_prompt = Path('prompts/qa.txt').read_text()
    
    prompt = ChatPromptTemplate([
        ('system', system_prompt),
        ('human', qa_prompt)]
    )

    OutputParser = StrOutputParser()
    chain = prompt | llm | OutputParser
    return chain

def format_docs(docs):
    return '\n\n'.join(
        f'Source: {d.metadata.get('source', '')}\n {d.page_content}'
        for d in docs
    )