import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import requests
from bs4 import BeautifulSoup

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Website Chatbot", layout="centered")
st.title("🌐 Website Based Chatbot ")

# ----------------------------
# Load Website
# ----------------------------
def load_website(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    return soup.get_text(separator=" ")

# ----------------------------
# Sidebar
# ----------------------------
url = st.sidebar.text_input("Enter Website URL")
load_btn = st.sidebar.button("Load Website")

# ----------------------------
# Load & Process
# ----------------------------
if load_btn and url:
    with st.spinner("Loading website..."):
        text = load_website(url)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_texts(chunks, embeddings)

        # ----------------------------
        # LLM (FLAN-T5)
        # ----------------------------
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        st.session_state.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={
                "prompt": None
            }
        )

        st.success("Website loaded successfully!")

# ----------------------------
# Ask Question
# ----------------------------
if "qa" in st.session_state:
    question = st.text_input("Ask a question from the website")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            answer = st.session_state.qa.run(
                f"Answer the question clearly in 2-3 sentences only.\nQuestion: {question}"
            )
            st.markdown("### ✅ Answer")
            st.write(answer)







