# for retriver i need vecstore to be present
from vectorstore.store import vector_store

def get_retriver(k= 4):
    vec_store = vector_store
    retriver = vec_store.as_retriever(search_kwargs= {'k': k})

    return retriver, vec_store 


