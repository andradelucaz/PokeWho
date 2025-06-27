# %% 
# import streamlit as st
from langchain_core.messages import HumanMessage
import streamlit as st

st.title("🤖 PokeWho - Agente Pokemon")
st.write("Converse com o assistente Pokemon!")

# Cache do agente
@st.cache_resource
def load_agent():
    from agent import create_poke_agent
    return create_poke_agent()

# Input do usuário
user_input = st.text_input("Pergunte sobre Pokemon:", placeholder="Ex: Me fale sobre Pikachu")

if user_input:
    st.write(f"👤 **Você:** {user_input}")
    
    try:
        # Carregar agente
        agent = load_agent()
        
        # Criar state inicial
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "context": [],
            "answer": ""
        }
        
        # Invocar agente
        with st.spinner("🔍 Pensando..."):
            result = agent.invoke(initial_state)
        
        # Mostrar resposta
        final_message = result["messages"][-1]
        st.write(f"🤖 **PokeWho:** {final_message.content}")
        
        # Debug info (opcional)
        with st.expander("Debug Info"):
            st.write("**Messages:**", len(result["messages"]))
            st.write("**Final State:**", {k: v for k, v in result.items() if k != "messages"})
            
    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")
        st.write("**Debug:**", type(e).__name__)

# Separador
st.write("---")
st.write("**Teste dos Retrievers Individuais (para debug):**")

# # Manter testes individuais para debug
# col1, col2 = st.columns(2)

# with col1:
#     if st.button("Testar Retriever Texto"):
#         try:
#             from retriever import poke_text_retriever
#             results = poke_text_retriever.invoke({"query": "pikachu"})
#             st.write("✅ Texto OK:", len(results), "resultados")
#         except Exception as e:
#             st.error(f"❌ Texto: {e}")

# with col2:
#     if st.button("Testar Retriever Imagem"):
#         try:
#             from retriever import poke_img_retriever
#             results = poke_img_retriever.invoke({"query": "pikachu"})
#             st.write("✅ Imagem OK:", len(results), "resultados")
#         except Exception as e:
#             st.error(f"❌ Imagem: {e}")
# %%
