#%% 
import pokebase as pb
import requests

# Getting Pokemon Description

def get_pokemon_description(pokemon_input):  

    pokemon = pb.pokemon(pokemon_input)

    # Basic charcteristics
    name = pokemon.name
    habitat = pokemon.species.habitat.name

    ability_list = []
    for a in pokemon.abilities:
        ability_list.append(a.ability.name) 

    poke_type = []
    for t in pokemon.types:
        poke_type.append(t.type.name)

    # Evolution Chain
    def get_evolution_chain(chain, result=None):
        if result is None:
            result = []
        result.append(chain['species']['name'])
        for evo in chain['evolves_to']:
            get_evolution_chain(evo, result)
        return result

    evolution_url = pokemon.species.evolution_chain.url
    evolution_data = requests.get(evolution_url).json()
    chain = evolution_data['chain']

    full_evolution_chain = get_evolution_chain(chain)

    sprites = pb.SpriteResource('pokemon', pokemon.id, official_artwork = True)
    img = sprites.img_data

    # Description 
    description = ( 
        f"{name.capitalize()} is a {', '.join(poke_type)} type Pokémon, who lives in {habitat}. "
        f"Abilities: {', '.join(ability_list)}. "
        f"Evolution Chain: {', '.join(full_evolution_chain)}" 
    )

    return {'description': description, 
            'name': name, 
            'habitat': habitat, 
            'ability': ability_list, 
            'poke_type':poke_type, 
            'evolution chain': full_evolution_chain}
#%%
from PIL import Image
from io import BytesIO
def get_pokemon_img(pokemon_input):

    sprites = pb.SpriteResource('pokemon', pokemon_input, official_artwork = True)
    img = sprites.img_data
    poke_img = Image.open(BytesIO(img))

    return [poke_img, img]

#%%
# Text Embedding 

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

model = SentenceTransformer('cenfis/turemb_512') 

import chromadb 
client = PersistentClient(path="./data/chromadb")
# collection = client.get_collection('pokemons')
# img_collection = client.get_collection('pokemon_images')
collection = client.create_collection('pokemons')
img_collection = client.create_collection('pokemon_images')

from img2vec_pytorch import Img2Vec
#%%
for i in range(151, 302):

    # Getting descriptions
    result = get_pokemon_description(i)

    # Create embeddings:
    text_embedding = model.encode(result['description'])

    # Create metadata for the vectordb
    description = result.get('description') or ""

    metadata = {"name": result['name'],
         "habitat": result["habitat"],
         "abilities": ", ".join(result['ability']),
         "type": ", ".join(result['poke_type']),
         "evolution": ", ".join(result["evolution chain"]),
         "page_content": description}

    # Armazenando na vectordb 
    collection.add(
        embeddings = [text_embedding],
        metadatas = [metadata],
        ids = [f"text-{i}"],
        documents = [description]
    )

    # Getting poke img 
    poke_img = get_pokemon_img(i)

    # Create img embeddings
    img2vec = Img2Vec(cuda=False)
    vector = img2vec.get_vec(poke_img[0])

    img_path = f"sprites/{result['name']}.png"
    with open(img_path, "wb") as f:
        f.write(poke_img[1])


    #Create img metadata
    img_metadata =  {"name": result['name'],
         "habitat": result["habitat"],
         "abilities": ", ".join(result['ability']),
         "type": ", ".join(result['poke_type']),
         "evolution": ", ".join(result["evolution chain"]),
         "img": img_path,
         "page_content": description}

    #Armazenando o img_embedding na vectordb 
    img_collection.add(
        embeddings = [vector],
        metadatas = [img_metadata],
        ids = [f"img-{i}"],
        documents = [description]
    )

 # %%
