import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from openai import OpenAI


# Define constants for file paths
INDEX_DIR = "faiss_index"
PDF_PATH = "faq.pdf"
load_dotenv()

def load_docs_from_pdf(pdf_path=PDF_PATH):
    """Load and split the PDF into chunks."""
    start = time.time()
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=150)
    processed_docs = splitter.split_documents(docs)
    st.write(f"{os.listdir()}")
    st.write(f"✅ Documents loaded from {pdf_path}")
    st.write(f"⏱ Time taken: {time.time() - start:.2f} seconds")
    return processed_docs

def build_or_load_vector_db(documents):
    """Build FAISS index or load from disk if already built."""
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"trust_remote_code": True}
    )

    if os.path.exists(INDEX_DIR):
        st.write("📂 Loading existing FAISS index from disk…")
        return FAISS.load_local(INDEX_DIR, embedder, allow_dangerous_deserialization=True)
    
    st.write("⚙️ Building new FAISS index…")
    start = time.time()
    db = FAISS.from_documents(documents, embedder)
    db.save_local(INDEX_DIR)
    return db

def main():
    st.set_page_config(page_title='E-Therapy Chatbot')
    st.title("FAQ Chatbot")

    # Init conversation state
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "document_db" not in st.session_state:
        with st.spinner("Loading FAQ and creating/loading vector store…"):
            docs = load_docs_from_pdf(PDF_PATH)
            st.session_state.document_db = build_or_load_vector_db(docs)

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key= os.getenv("OPENAI_API_KEY"),
    )

    def submit_with_db():
        user_input = st.session_state.query_input.strip()
        if not user_input:
            return
        
        # Add user message to conversation history
        st.session_state.conversation.append({'role': 'user', 'content': user_input})
        
        retriever = st.session_state.document_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 2}
        )
        results = retriever.invoke(user_input)
        
        # Clean text from retrieved docs
        context_text = "\n\n".join(doc.page_content for doc in results)

        # Construct a concise prompt for the LLM without full chat history
        constructed_prompt = f"""
        You are a helpful assistant. Answer the user's question based on the context below.
        Respond ONLY with information directly related to the context, and do not include any additional or irrelevant messages.
        DO NOT RESPOND WITH INFORMATION OUTSIDE THE PROVIDED CONTEXT.
        If the answer is not found in the context, respond with "I cannot help you with that based on my knowledge".

        Context: {context_text}
        Question: {user_input}
        Answer:
        """
        
        try:
            completion = client.chat.completions.create(
                model="mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
                messages=[
                    {
                        "role": "user",
                        "content": constructed_prompt
                    }
                ],
            )
        except Exception as err:
            st.error(f"Error during chat: {err}")
            reply = "An error occurred while fetching a response."

        # Add assistant response to conversation history
        if (completion):
            st.session_state.conversation.append({'role': 'assistant', 'content': completion.choices[0].message.content})
        else:
            st.session_state.conversation.append({'role': 'assistant', 'content': "An error occured"})
        st.session_state.query_input = ""
        # st.rerun()


    # Display chat history in reverse order
    for entry in st.session_state.conversation:
        with st.chat_message(entry['role']):
            st.write(entry['content'])
    st.text_area("Enter your question:", key="query_input")
    st.button("Submit", on_click=submit_with_db)

if __name__ == "__main__":
    main()
