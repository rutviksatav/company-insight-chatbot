import streamlit as st
import os
import logging
from dotenv import load_dotenv
from utils import get_rag_chain

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit config
st.set_page_config(page_title="Company Insights Chatbot", layout="centered")

# Title and New Chat Button
# Title and New Chat Button
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown("## 💼 Company Insights Chatbot")  # Smaller and better aligned title
with col2:
    new_chat = st.button("🧹 New Chat", use_container_width=True)
    if new_chat:
        logger.info("New chat session initiated")
        st.session_state.chat_history = []
        st.rerun()


st.caption("Ask about companies, financial metrics, sectors, or business insights")

# Load RAG chain
@st.cache_resource
def init_rag():
    try:
        logger.info("Initializing RAG chain from cache")
        rag_chain = get_rag_chain()
        logger.info("RAG chain loaded successfully")
        return rag_chain
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}", exc_info=True)
        st.error("Failed to initialize the chatbot. Please try again later.")
        raise

rag_chain = init_rag()

# Chat History Init
if 'chat_history' not in st.session_state:
    logger.info("Initializing chat history in session state")
    st.session_state.chat_history = []

# Input Validation
def validate_query(query):
    if not query or query.strip() == "":
        logger.warning("Empty query received")
        return False, "Please enter a valid question."
    if len(query) > 500:
        logger.warning("Query too long")
        return False, "Query is too long. Please keep it under 500 characters."
    return True, ""

# Chat UI
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.write(message["content"])
        if message['role'] == 'assistant' and 'sources' in message:
            for i, doc in enumerate(message['sources']):
                with st.expander(f"Source {i+1}: {doc.metadata['company']} ({doc.metadata['symbol']})"):
                    st.write(f"**UUID**: {doc.metadata['uuid']}")
                    st.write(f"**Content**: {doc.page_content}")

# Chat Input
query = st.chat_input("Ask about companies, metrics, or industries...")

# Process query
if query:
    logger.info(f"Received user query: {query}")
    is_valid, error_message = validate_query(query)
    if not is_valid:
        st.error(error_message)
        logger.warning(f"Invalid query: {error_message}")
    else:
        try:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.spinner("Processing..."):
                result = rag_chain.invoke({"query": query})
                answer = result["result"]
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
            })
            logger.info("Query processed successfully")
            st.rerun()
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            st.error("An error occurred while processing your request. Please try again.")
