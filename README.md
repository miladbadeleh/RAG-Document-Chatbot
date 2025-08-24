# R-Powered Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the content of any uploaded PDF document.

## 🚀 Features

*   **Document Processing:** Extracts and chunks text from uploaded PDFs.
*   **Semantic Search:** Uses OpenAI embeddings and ChromaDB to find relevant text passages.
*   **Context-Aware Answers:** Leverages GPT-3.5-turbo to generate answers based *only* on the provided document context.
*   **Citations:** Shows the exact source passages used to generate each answer, ensuring transparency.
*   **Web UI:** Built with Streamlit for an intuitive user experience.

## 🛠️ Tech Stack

*   **Framework:** LangChain
*   **Embeddings:** OpenAI `text-embedding-ada-002`
*   **Vector Database:** ChromaDB
*   **LLM:** OpenAI `gpt-3.5-turbo`
*   **Web App:** Streamlit

## 📦 Installation & Usage

1.  Clone the repo and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Get an [OpenAI API key](https://platform.openai.com/api-keys) and add it to a `.env` file:
    ```env
    OPENAI_API_KEY='your-key-here'
    ```
3.  Run the application:
    ```bash
    streamlit run app.py
    ```
4.  Upload a PDF and start asking questions!

## 🎯 How It Works (RAG Architecture)

1.  **Load:** The PDF text is extracted.
2.  **Split:** The text is split into manageable chunks.
3.  **Embed:** Each chunk is converted into a vector (embedding) using OpenAI.
4.  **Store:** Vectors are stored in ChromaDB.
5.  **Retrieve:** When a question is asked, its embedding is compared to all stored vectors to find the most relevant chunks.
6.  **Generate:** The question and retrieved chunks are sent to the LLM with instructions to answer based solely on the context.

## 🔮 Future Improvements

*   Add support for other file types (DOCX, PPTX, TXT).
*   Implement conversational memory so the chatbot can remember previous questions in a session.
*   Deploy the app publicly on Streamlit Community Cloud or Hugging Face Spaces.
