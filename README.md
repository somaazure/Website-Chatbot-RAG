# Website-Chatbot-RAG
Website Chatbot-RAG with Dynamic URL

Use this command to run in a virtual env: streamlit run Chatbot-memory_based-DynUrl-St.py

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

Example URL Tried: https://lilianweng.github.io/posts/2023-06-23-agent/
Below is the Screen Shot.

![image](https://github.com/user-attachments/assets/f6369f3d-ec23-4588-8038-0ce4d5f1032c)

