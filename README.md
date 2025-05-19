# Website-Chatbot-RAG
Website Chatbot-RAG with Dynamic URL


**🔍 Website-Based Chatbot – Core Functionality:**

1. Accepts a public website URL from the user via a Streamlit web interface.
2. Fetches and scrapes readable content from the webpage using `requests` and `BeautifulSoup`.
3. Cleans and splits the content into manageable text chunks using LangChain’s `RecursiveCharacterTextSplitter`.
4. Converts text chunks into vector embeddings using OpenAI's `text-embedding` model.
5. Stores these embeddings in a local FAISS vector database for efficient similarity search.
6. Uses `ChatOpenAI` (GPT-4o-mini) as the LLM for answering user queries.
7. Enables conversational memory with `ConversationBufferMemory` for context-aware responses.
8. Matches user queries to relevant chunks using semantic similarity retrieval.
9. Constructs prompts with contextual snippets and sends them to the LLM.
10. Returns accurate, context-based answers about the website content in a chat-like format.

![image](https://github.com/user-attachments/assets/9e911afe-3188-43a1-9800-48197dda403d)
