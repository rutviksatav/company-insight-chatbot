Company Insights Chatbot
Overview
A Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that acts as a friendly financial analyst assistant. It answers queries about company data, financial metrics, sectors, and industries using vector search, embeddings, and a large language model (LLM).
Features

Friendly, interactive UI with chat history and source details.
Provides company info: name, exchange, sector, financials (price, market cap, etc.), employees, location, and business summary.
Logs errors for debugging.
Securely loads credentials via environment variables.
Supports "New Chat" to reset conversation.
Displays source documents for transparency.

Requirements

Python 3.8+
Dependencies (install via pip install -r requirements.txt):
streamlit
langchain_community
langchain_groq
langchain_core
python-dotenv
faiss-cpu
sentence-transformers
groq


Environment variables:
HUGGINGFACE_TOKEN: HuggingFace model access token.
GROQ_API_KEY: Groq LLM API key.



Setup

Clone the Repository:
git clone <repository-url>
cd <repository-directory>


Install Dependencies:
pip install -r requirements.txt


Set Up Environment Variables: Create a .env file in the project root:
HUGGINGFACE_TOKEN=<your-huggingface-token>
GROQ_API_KEY=<your-groq-api-key>


Prepare FAISS Index: Ensure a faiss_index directory exists with pre-indexed company data compatible with the sentence-transformers/all-MiniLM-L6-v2 model.


Usage

Run the Application:
streamlit run main.py

This launches the Streamlit app, which:

Loads embeddings (sentence-transformers/all-MiniLM-L6-v2).
Loads the FAISS vector store.
Uses Groq LLM (llama3-8b-8192).
Sets up the RetrievalQA chain with a custom prompt.


Interact with the Chatbot:

Open the app in your browser.
Ask about companies (e.g., "What are Apple's financials?") or say "hello" for a greeting.
View answers, chat history, and expandable source details.
Click "New Chat" to reset the conversation.



Code Structure

Main Script (main.py): Streamlit app with UI, chat logic, and query processing.
Utils (utils.py): Contains RAG chain setup (embeddings, vector store, LLM, and prompt).
Logging: Outputs to app.log and console (INFO for operations, ERROR for issues).
Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2.
Vector Store: FAISS for fast similarity search.
LLM: Groq llama3-8b-8192 (temperature 0.4).
Prompt: Ensures friendly, concise answers.
RAG Chain: Combines retrieval and generation (top 5 results).

Notes

FAISS index must be pre-generated.
Set environment variables to avoid errors.
Queries are validated (non-empty, <500 characters).
If data is missing, general industry insights are provided.

License
MIT License. See LICENSE file for details.
