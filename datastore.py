#%% 
import pokebase as pb
import requests


pokemon = pb.pokemon('pikachu')

name = pokemon.name

base_experience = pokemon.base_experience
habitat = pokemon.species.habitat.name if pokemon.species.habitat else "unknown"
    
# Getting Pokemon Description
#%%
def get_pokemon_data(pokemon_input):  

    pokemon = pb.pokemon(pokemon_input)

    # Basic charcteristics
    name = pokemon.name

    base_experience = pokemon.base_experience

    habitat = pokemon.species.habitat.name if pokemon.species.habitat else "unknown"
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

    # Catch rate / is_legendary / is_mythical
    catch_rate = pokemon.species.capture_rate
    is_legendary = pokemon.species.is_legendary
    is_mythical = pokemon.species.is_mythical

    # Num of evolution
    evolution_chain_length = len(full_evolution_chain)
    # Gen
    generation = pokemon.species.generation.name

    # Base Stats
    base_stats = { stat.stat.name: stat.base_stat for stat in pokemon.stats } 

    # Pokemon Description (GAme based) 
    def get_description(pokemon_name):
        try:
            species = pb.pokemon_species(pokemon_name)
            description = next(
                entry.flavor_text for entry in species.flavor_text_entries
                if entry.language.name == 'en'
            )
            return description.replace('\n', ' ').replace('\f', ' ')
        except:
            return "No description available"
    
    pokemon_description = get_description(name)

    sprites = pb.SpriteResource('pokemon', pokemon.id, official_artwork = True)
    img = sprites.img_data

    # Description 
    content = (
    f"{name.capitalize()} is a Generation {generation[-1]} Pokémon. "
    f"It is of type {', '.join(poke_type)} and has the following abilities: {', '.join(ability_list)}. "
    f"{name.capitalize()} typically inhabits {habitat}. "
    f"It has base stats including {', '.join([f'{k}: {v}' for k, v in base_stats.items()])}. "
    f"It {'is' if is_legendary else 'is not'} considered a legendary Pokémon and "
    f"{'is' if is_mythical else 'is not'} a mythical Pokémon. "
    f"It has a capture rate of {catch_rate} and belongs to an evolution chain of {evolution_chain_length} Pokémon: "
    f"{' > '.join(full_evolution_chain)}. "
    f"Description: {pokemon_description}"
)


    return {
    'name': name, 
    'habitat': habitat, 
    'ability': ability_list, 
    'poke_type': poke_type, 
    'evolution chain': full_evolution_chain,
    'base_stats': base_stats,
    'generation': generation,
    'evolution_chain_length': evolution_chain_length,
    'catch_rate': catch_rate,
    'is_legendary': is_legendary,
    'is_mythical': is_mythical,
    "pokemon_description": pokemon_description,
    "content": content
    }


#%%
# from PIL import Image
# from io import BytesIO
# def get_pokemon_img(pokemon_input):

#     sprites = pb.SpriteResource('pokemon', pokemon_input, official_artwork = True)
#     img = sprites.img_data
#     poke_img = Image.open(BytesIO(img))

#     return [poke_img, img]

#%%
# Text Embedding 

from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient

model = SentenceTransformer('all-MiniLM-L6-v2') 

import chromadb 
client = PersistentClient(path="./data/chromadb")
#collection = client.get_collection('pokemons')
# img_collection = client.get_collection('pokemon_images')
collection = client.create_collection('pokemons')
#img_collection = client.create_collection('pokemon_images')

from img2vec_pytorch import Img2Vec

#%%
for i in range(832, 1032):

    # Getting descriptions
    result = get_pokemon_data(i)

    # Create embeddings:
    text_embedding = model.encode(result['content'])

    # Create metadata for the vectordb
    description = result.get('content') or ""

    metadata = {
    "name": result['name'],
    "habitat": result["habitat"],
    "abilities": ", ".join(result['ability']),
    "type": ", ".join(result['poke_type']),
    "evolution": ", ".join(result["evolution chain"]),
    "generation": result["generation"],
    "evolution_chain_length": result["evolution_chain_length"],
    "catch_rate": result["catch_rate"],
    "is_legendary": result["is_legendary"],
    "is_mythical": result["is_mythical"],
    "base_hp": result["base_stats"]["hp"],
    "base_attack": result["base_stats"]["attack"],
    "base_defense": result["base_stats"]["defense"],
    "base_sp_attack": result["base_stats"]["special-attack"],
    "base_sp_defense": result["base_stats"]["special-defense"],
    "base_speed": result["base_stats"]["speed"],
    "pokemon_description": result["pokemon_description"],
    "page_content": description
}


    # Armazenando na vectordb 
    collection.add(
        embeddings = [text_embedding],
        metadatas = [metadata],
        ids = [f"text-{i}"],
        documents = [description]
    )

    # # Getting poke img 
    # poke_img = get_pokemon_img(i)

    # # Create img embeddings
    # img2vec = Img2Vec(cuda=False)
    # vector = img2vec.get_vec(poke_img[0])

    # img_path = f"sprites/{result['name']}.png"
    # with open(img_path, "wb") as f:
    #     f.write(poke_img[1])


    # #Create img metadata
    # img_metadata =  {"name": result['name'],
    #      "habitat": result["habitat"],
    #      "abilities": ", ".join(result['ability']),
    #      "type": ", ".join(result['poke_type']),
    #      "evolution": ", ".join(result["evolution chain"]),
    #      "img": img_path,
    #      "page_content": description}

    # #Armazenando o img_embedding na vectordb 
    # img_collection.add(
    #     embeddings = [vector],
    #     metadatas = [img_metadata],
    #     ids = [f"img-{i}"],
    #     documents = [description]
    # )

    print(result['name'],"(",i,")", " loaded")


