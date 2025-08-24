# app.py
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# app.py (continued)
@st.cache_resource(show_spinner=False) # Cache this resource so it only runs once
def process_document(uploaded_file):
    """
    Takes an uploaded PDF file, extracts text, chunks it, and creates a vector database.
    """
    with st.spinner("Processing document... This may take a moment."):
        # 1. Extract text from PDF
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or "" # Handle pages with no text

        # 2. Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000, # Size of each chunk
            chunk_overlap=200, # Overlap to avoid losing context
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # 3. Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = Chroma.from_texts(texts=chunks, embedding=embeddings)

        # 4. Create a retriever from the vector store
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4} # Return top 4 most relevant chunks
        )
        return retriever



# app.py (continued)
def setup_qa_chain(retriever):
    """
    Sets up the question-answering chain with a custom prompt.
    """
    # Custom prompt to guide the LLM
    prompt_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Be polite and professional.

    {context}

    Question: {question}
    Helpful Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Choose the LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0, # Low temperature for factual, deterministic answers
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" is simplest: stuffs all context into the prompt
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True # Crucial for showing citations!
    )
    return qa_chain




# app.py (continued)
def main():
    st.title("📄 Document Q&A Chatbot")
    st.write("Upload a PDF and ask questions about its content!")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Initialize the retriever and QA chain
        retriever = process_document(uploaded_file)
        qa_chain = setup_qa_chain(retriever)

        # Initialize chat history in Streamlit's session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # If it's an assistant message, also show sources
                if message.get("sources"):
                    with st.expander("Source Citations"):
                        st.write(message["sources"])

        # React to user input
        if question := st.chat_input("Ask a question about the document:"):
            # Display user message in chat message container
            st.chat_message("user").markdown(question)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": question})

            # Get answer from QA chain
            with st.spinner("Thinking..."):
                result = qa_chain({"query": question})
                answer = result["result"]
                sources = result["source_documents"]

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Source Citations"):
                    for i, doc in enumerate(sources):
                        st.write(f"**Source {i+1}:** (Page ~{doc.metadata.get('page', 'N/A')})")
                        st.caption(doc.page_content[:500] + "...") # Show first 500 chars

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

if __name__ == "__main__":
    main()
