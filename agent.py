#%%
from typing import TypedDict, List, Literal, Annotated
from retriever import  poke_text_retriever 
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain.tools import Tool 
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

# Criando State
class PokemonState(TypedDict):
    #input_type: Literal['text', 'image']
    context: List[Document] 
    answer: str
    messages: Annotated[list[AnyMessage], add_messages]

def create_poke_agent():
    # Initialize tools

    # img_search = poke_img_retriever() 
    text_search = poke_text_retriever

    # Generate the chat interface, including the tools

    tools = [text_search] 

    model = ChatOpenAI(
        model = "gemma-3-12b-it",
        openai_api_base = 'http://127.0.0.1:1234/v1',
        openai_api_key = 'not-needed'
    )

    chat_with_tools = model.bind_tools(tools)


    # Generate the AgentState and Agent graph 

    def assistant(state: PokemonState):
        response = chat_with_tools.invoke(state['messages'])
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

