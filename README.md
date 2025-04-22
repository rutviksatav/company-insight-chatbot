# Company Insights Chatbot

A Streamlit-based chatbot that provides insights about companies, including financial metrics, sectors, industries, and business summaries, using a Retrieval-Augmented Generation (RAG) pipeline.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Environment Variables](#environment-variables)
- [Logging](#logging)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Company Insights Chatbot is a web application built with Streamlit, leveraging a RAG pipeline to answer user queries about companies. It uses a FAISS vector store for document retrieval, HuggingFace embeddings for text processing, and the Groq API for language model inference. The chatbot is designed to provide concise, company-specific insights based on pre-indexed data.

## Features
- **Interactive Chat Interface**: Users can ask questions about companies, financial metrics, sectors, and industries.
- **Example Prompts**: Displays sample questions to guide users (e.g., "What is the market cap of Microsoft?").
- **Chat History**: Maintains conversation history within a session, with a "New Chat" button to reset.
- **Source Attribution**: Shows source documents for answers, including company details and metadata.
- **Input Validation**: Ensures queries are valid and within length limits (500 characters).
- **Error Handling**: Logs errors and displays user-friendly messages for issues like initialization failures.
- **Logging**: Comprehensive logging to both file (`app.log`) and console for debugging.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd company-insights-chatbot
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add the following:
   ```
   HUGGINGFACE_TOKEN=<your-huggingface-token>
   GROQ_API_KEY=<your-groq-api-key>
   ```

5. **Prepare FAISS Index**:
   Ensure the FAISS vector store (`faiss_index`) is available in the project directory. This should contain pre-indexed company data.

## Usage
1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the Chatbot**:
   - Open the app in your browser (typically at `http://localhost:8501`).
   - Enter queries like "Tell me about Apple Inc." or "What is the market cap of Microsoft?".
   - Use the "New Chat" button to start a fresh session.
   - Type "help" or "examples" to see sample prompts.

3. **View Logs**:
   Check `app.log` for detailed logs of application events, errors, and queries.

## Project Structure
```
company-insights-chatbot/
├── app.py              # Main Streamlit application
├── utils.py            # Utility functions for RAG pipeline setup
├── faiss_index/        # FAISS vector store directory
├── app.log             # Log file
├── .env                # Environment variables
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Dependencies
- `streamlit`: Web app framework
- `langchain`: RAG pipeline and LLM integration
- `langchain_community`: FAISS vector store and HuggingFace embeddings
- `langchain_groq`: Groq API integration
- `python-dotenv`: Environment variable management
- `logging`: Logging functionality

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Environment Variables
- `HUGGINGFACE_TOKEN`: API token for HuggingFace embeddings.
- `GROQ_API_KEY`: API key for Groq language model.

Ensure these are set in the `.env` file or your environment.

## Logging
- Logs are written to `app.log` and printed to the console.
- Log levels include `INFO`, `WARNING`, and `ERROR`.
- Key events logged: application startup, query processing, errors, and RAG chain initialization.

## Limitations
- **Data Dependency**: Answers are limited to the data in the FAISS vector store.
- **Query Scope**: Only responds to company-specific questions; other queries return a fallback message.
- **Character Limit**: Queries are capped at 500 characters.
- **Offline Operation**: Requires internet access for HuggingFace and Groq APIs.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
