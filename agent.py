#%% 
import os
import time
from typing import TypedDict, List, Literal, Annotated
from dotenv import load_dotenv
from retriever import poke_text_retriever 
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain.tools import Tool 
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

# Carregar variáveis do arquivo .env
load_dotenv()

# Criando State
class PokemonState(TypedDict):
    #input_type: Literal['text', 'image']
    context: List[Document] 
    answer: str
    messages: Annotated[list[AnyMessage], add_messages]

# System prompt para melhorar as respostas
POKEMON_SYSTEM_PROMPT = """You are PokeWho, a highly knowledgeable and friendly Pokémon expert.

INSTRUCTIONS:
    - When a user asks about a Pokémon, FIRST extract the Pokémon name from their question
    - ALWAYS use the poke_text_retriever tool with the parameter "query" containing the Pokémon name
      Example: If user asks "Tell me about Pikachu", you must call: poke_text_retriever(query="Pikachu")
    - Craft complete and engaging answers using the data you find
    - Organize information in a clear and captivating way
    - Include details about: types, abilities, stats, habitat, evolution, generation
    - Add fun facts and context when appropriate
    - Use emojis to make the answers more visual
    - If no data is found, be honest but provide general information you know

TOOL USAGE:
    - Tool name: poke_text_retriever
    - Parameter name: query (not pokemon_name)
    - Example usage: poke_text_retriever(query="Charmander")
    
RESPONSE FORMAT:
    - Start with a catchy introduction about the Pokémon
    - Organize the information into clear sections
    - End with an interesting tip or fun fact
    
TONE EXAMPLE:
    "🔥 Charmander is truly special! This adorable fire lizard..."
    
Always be educational, friendly, and enthusiastic about Pokémon!"""

model = ChatOpenAI(
    model="qwen2.5-7b-instruct-1m",
    openai_api_base='http://127.0.0.1:1234/v1',
    openai_api_key='not-needed',
    temperature=0.7
)

def create_poke_agent():
    # Initialize tools
    text_search = poke_text_retriever
    tools = [text_search] 

    # Generate the chat interface, including the tools
    chat_with_tools = model.bind_tools(tools)

    # Generate the AgentState and Agent graph 
    def assistant(state: PokemonState):
        start_time = time.time()
        print(f"🤖 [ASSISTANT] Iniciando processamento...")
        
        # Adicionar system prompt às mensagens
        from langchain_core.messages import SystemMessage
        messages = state['messages']
        
        # Se não há system message, adicionar
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=POKEMON_SYSTEM_PROMPT)] + messages
        
        response = chat_with_tools.invoke(messages)
        
        end_time = time.time()
        latency = end_time - start_time
        print(f"🤖 [ASSISTANT] Concluído em {latency:.2f}s")
        
        return {"messages": [response]}
        
    # Inicializando o grafo 
    graph = StateGraph(PokemonState)
    graph.add_node('assistant', assistant)
    graph.add_node('tools', ToolNode(tools))

    # Definindo fluxo 
    graph.add_conditional_edges(
        "assistant",
        #If the latest message requires a tool, route to tools
        #Otherwise, provide a direct response
        tools_condition
    )

    graph.add_edge("tools", "assistant") 
    graph.set_entry_point("assistant")

    poke_agent = graph.compile() 
    return poke_agent