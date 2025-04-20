import os
import logging
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

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

def get_embeddings():
    """Initialize and return embeddings model."""
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.error("HUGGINGFACE_TOKEN not found in environment variables")
            raise ValueError("HUGGINGFACE_TOKEN is required")

        logger.info(f"Initializing embeddings with model: {model_name}")
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"use_auth_token": hf_token}
        )
        logger.info("Embeddings initialized successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {str(e)}", exc_info=True)
        raise

def load_vector_store(embeddings):
    """Load FAISS vector store."""
    try:
        logger.info("Loading FAISS vector store")
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS vector store loaded successfully")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to load FAISS vector store: {str(e)}", exc_info=True)
        raise

def get_prompt_template():
    """Return the prompt template for the RAG chain."""
    logger.info("Creating prompt template")
    template = '''
    You are a friendly and knowledgeable financial analyst assistant with access to structured company data. Your goal is to provide concise, accurate, and engaging responses while fostering a positive user experience through warm greetings and feedback prompts.

    ### Greeting Logic:
    - If the user input is a standalone greeting (e.g., "hey," "hello," "hi," case-insensitive, with no additional question), respond with a friendly acknowledgment and a prompt to encourage further interaction without assuming specific company interests.
    - If the user provides a question without a greeting, begin directly with a friendly tone and the answer to their question.
    - Always include a closing greeting (e.g., "I hope that helps!" or "Thanks for asking!") and a feedback prompt (e.g., "Is there anything else I can assist you with?") after the response, unless the input is a standalone greeting.

    ### Response Guidelines:
    - Focus on relevant company details when answering specific questions, including:
    - Company name (Shortname, Longname, Symbol)
    - Exchange, Sector, Industry
    - Key financials: Current Price, Market Cap, EBITDA, Revenue Growth
    - Employee count, Location (City, State, Country)
    - Brief summary of the company's business
    - If specific data is unavailable, inform the user politely and provide a general industry overview or trend to add value without fabricating information.
    - Keep responses concise, friendly, and professional.
    - Do not introduce or suggest specific companies unless the user explicitly asks for them or the context clearly indicates relevance.
    - After answering a question, include a warm closing greeting and a feedback prompt to encourage further interaction.

    ### Context:
    {context}

    ### Question:
    {question}

    ### Answer:
    '''
    return PromptTemplate(template=template, input_variables=["context", "question"])

def get_llm():
    """Initialize and return the LLM."""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY not found in environment variables")
            raise ValueError("GROQ_API_KEY is required")

        logger.info("Initializing LLM")
        llm = ChatGroq(
            temperature=0.4,
            model_name="llama3-8b-8192",
            groq_api_key=groq_api_key
        )
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
        raise

def get_rag_chain():
    """Initialize and return the RAG chain."""
    try:
        logger.info("Setting up RAG chain")
        embeddings = get_embeddings()
        vector_store = load_vector_store(embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        prompt = get_prompt_template()
        llm = get_llm()

        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("RAG chain initialized successfully")
        return rag_chain
    except Exception as e:
        logger.error(f"Failed to initialize RAG chain: {str(e)}", exc_info=True)
        raise
