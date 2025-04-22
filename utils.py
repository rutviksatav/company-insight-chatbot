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
    You are a helpful and concise assistant that only answers company-specific insight questions based on the provided context.

    ## Greeting Guideline:
    - If user greets, greet back with: "Hello! How can I help you?"

    ## Answering Guidelines:
    - ONLY respond to questions related to company insights based on the context.
    - Use these key data points if available in the context:
    - Company Name, Symbol, Exchange
    - Sector, Industry, Market Cap, Revenue, EBITDA
    - Current Price, Growth Metrics
    - HQ Location, Number of Employees
    - Business Summary
    - If data is missing, respond with: "I don't have information at the moment."

    ## Do Not Answer:
    - Do NOT respond to questions that are:
    - Explicit or inappropriate in nature.
    - Unrelated to the context or company insights.
    - Code or programming-related (e.g., "write code", "how to code", "give me a script").
    - Generic factual questions not found in the context.
    - Don't add "Hello! How can I help you?" in every answer.
    - For such questions, respond with:
    "I can't help with that. I can only help you with company insights."

    ## Response Format:
    - Only answer based on the given context.
    - If the answer is not in the context, say: "I don't have information. I can only help you with the company insight."
    - Summarize the answer concisely and clearly.

    Context:
    {context}

    Question:
    {question}

    Answer:
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
            temperature=0.2,
            model_name="llama3-8b-8192",
            groq_api_key=groq_api_key,
            streaming = True
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
