#%%
# Create a retriever (LangChain Style)
from langchain.tools import tool
from langchain_community.vectorstores import Chroma 
#from langchain_community.embeddings import SentenceTransformerEmbeddings 
from chromadb import PersistentClient
from typing import List


@tool
def poke_text_retriever(query: str) -> List[str]:
    """"
    Searches for detailed information about Pokémon in the database.

    This tool returns comprehensive information including:
        - Pokémon name and type
        - Abilities and base stats (HP, Attack, Defense, etc.)
        - Natural habitat and generation
        - Full evolution chain
        - Legendary/Mythical status
        - Capture rate
        - Official in-game description

    Always use this tool when asked about any specific Pokémon."""
    
    import time
    start_time = time.time()
    print(f"🔍 [RETRIEVER] Buscando por: '{query}'")
    
    from langchain_core.documents import Document
    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer 

    # Initialize Chorma vectorstore 
    db_start = time.time()
    client = PersistentClient(path="./data/chromadb")
    collection = client.get_collection('pokemons') 
    print(f"🗄️ [CHROMADB] Conectado em {time.time() - db_start:.2f}s")

    #Create query embedding 
    embed_start = time.time()
    
    # Cache do modelo embedding para evitar recarregamento
    global _embedding_model
    if '_embedding_model' not in globals():
        print("📥 [EMBEDDING] Carregando modelo pela primeira vez...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ [EMBEDDING] Modelo carregado e cacheado!")
    
    query_embedding = _embedding_model.encode(query)
    print(f"🧠 [EMBEDDING] Gerado em {time.time() - embed_start:.2f}s")

    #Search in the vectordb 
    search_start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    print(f"🔎 [SEARCH] Busca em {time.time() - search_start:.2f}s")

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
    
    total_time = time.time() - start_time
    print(f"🔍 [RETRIEVER] Total: {total_time:.2f}s - {len(documents)} docs encontrados")
    return documents


# @tool
# def poke_img_retriever(query: str) -> List[str]:
#     """Busca a imagem de um Pokémon na base de dados.
    
#     Esta ferramenta retorna o caminho para a imagem do Pokémon.
#     Use esta ferramenta quando perguntarem sobre a aparência de um Pokémon.
#     """
#     from chromadb import PersistentClient
#     from img2vec_pytorch import Img2Vec
#     from PIL import Image
#     from langchain_core.documents import Document

#     # Initialize Chorma vectorstore 
#     client = PersistentClient(path="./data/chromadb")
#     collection = client.get_collection('pokemon_images') 

#     # Create query embedding 
#     img2vec = Img2Vec(cuda=False)
#     img = Image.open(query)
#     query_embedding = img2vec.get_vec(img)

#     # Search in the vectordb 
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=1
#     )

#     # Create Documents 
#     documents = [] 
#     for i in range(len(results['documents'][0])):
#         texto = results['documents'][0][i] 
#         if texto:
#             doc = Document( 
#                 page_content = texto,
#                 metadata = results['metadatas'][0][i]
#             )
#             documents.append(doc)
    
#     return documents
# %%