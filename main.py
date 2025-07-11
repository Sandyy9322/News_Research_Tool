import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq  

load_dotenv()

st.title("RockyBot: News Research Tool ")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_groq.pkl"

main_placeholder = st.empty()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"  
)

if process_url_clicked:
    if not urls:
        st.warning("Please enter at least one valid URL.")
    else:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Loading data from URLs...")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(data)
        st.success(f"Split into {len(docs)} chunks ")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        main_placeholder.text("Creating vector store...")
        vectorstore = FAISS.from_documents(docs, embeddings)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("Vector store created and saved ")

# Search Interface
query = main_placeholder.text_input("Ask a question about the articles:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result.get("answer", "No answer found."))

        if result.get("sources"):
            st.subheader("Sources:")
            for src in result["sources"].split("\n"):
                st.write(src)
    else:
        st.warning("Please process URLs first.")
