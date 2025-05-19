import os
import requests
import bs4
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Set OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.text_input("🔑 Enter your OpenAI API key", type="password")
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Streamlit app config
st.set_page_config(page_title="Website Chatbot", layout="wide")
st.title("💬 Website Chatbot")
st.markdown("Ask questions about the content of any public website.")

# Input URL from user
website_url = st.text_input("🔗 Enter a website URL", placeholder="https://example.com")

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize model & embedding
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
embeddings = OpenAIEmbeddings()

# Prompt
prompt_template = """
You are an assistant that answers questions based on the provided website content.
Only use the information in the context below to answer.
If the context does not contain the answer, say "I don't know".

Context:
{context}

Human: {question}
Assistant:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Helper functions
def fetch_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.error(f"Error fetching the website: {e}")
        return None

def scrape_website(url):
    content = fetch_website_content(url)
    if not content:
        return []
    soup = bs4.BeautifulSoup(content, 'html.parser')
    text_content = soup.get_text(separator='\n', strip=True)
    return [text_content]

def process_website(url):
    docs = scrape_website(url)
    if not docs:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_text(docs[0])
    if not splits:
        return None
    return FAISS.from_texts(splits, embedding=embeddings)

# Chat interface
if website_url:
    with st.spinner("🔍 Scraping and processing website..."):
        vectorstore = process_website(website_url)

    if vectorstore:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_query = st.text_input("💬 Ask a question about the website", key="query_input")

        if user_query:
            result = qa_chain({"question": user_query})
            st.session_state.chat_history.append(("user", user_query))
            st.session_state.chat_history.append(("bot", result["answer"]))

        if st.session_state.chat_history:
            st.markdown("---")
            for role, message in st.session_state.chat_history:
                if role == "user":
                    st.markdown(f"**🧑 You:** {message}")
                else:
                    st.markdown(f"**🤖 Bot:** {message}")
    else:
        st.warning("❌ Unable to process website content. Try a different URL.")
