from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

def get_llm():
    model = HuggingFaceEndpoint(repo_id='MiniMaxAI/MiniMax-M2',
                                # task='text-generation'
                                )
    llm = ChatHuggingFace(llm = model)
    return llm