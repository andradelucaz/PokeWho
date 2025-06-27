#%%
# Create a retriever (LangChain Style)
from langchain.tools import tool
from langchain_community.vectorstores import Chroma 
#from langchain_community.embeddings import SentenceTransformerEmbeddings 
from chromadb import PersistentClient
from typing import List


@tool
def poke_text_retriever(query: str) -> List[str]:
    """Retrieve Pokemon description and other basic informations based on images"""
    
    from langchain_core.documents import Document
    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer 

    # Initialize Chorma vectorstore 
    client = PersistentClient(path="./data/chromadb")
    collection = client.get_collection('pokemons') 

    #Create query embedding 
    model = SentenceTransformer('cenfis/turemb_512')  
    query_embedding = model.encode(query)

    #Search in the vectordb 
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    # Create Documents 
    documents = [] 
    for i in range(len(results['documents'][0])):

        texto = results['documents'][0][i] 

        if texto:
            doc = Document( 
                page_content = texto,
                metadata = results['metadatas'][0][i]
            )
            documents.append(doc)
    return documents