import streamlit as st
import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter   # from classic
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_mistralai import MistralAIEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.retrievers import MultiQueryRetriever   # <-- using classic

load_dotenv()

# API keys
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") or st.secrets.get("MISTRAL_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not MISTRAL_API_KEY:
    st.error("MISTRAL_API_KEY not found. Please set it in .env or Streamlit secrets.")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in .env or Streamlit secrets.")

st.set_page_config(page_title="YouTube RAG Chat", layout="wide")
st.title("🎥 YouTube Video Chat with RAG")

@st.cache_resource
def get_embeddings():
    return MistralAIEmbeddings(model="mistral-embed")

@st.cache_resource
def get_llm():
    return ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

def load_transcript(video_id: str, language: str = "en") -> str:
    try:
        ytt_api = YouTubeTranscriptApi()
        transcripts = ytt_api.fetch(video_id, languages=[language])
        return " ".join(chunk.text for chunk in transcripts)
    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except Exception as e:
        raise RuntimeError(f"Error fetching transcript: {e}")

def get_vector_store(video_id: str, language: str, transcript_text: str = None):
    persist_dir = f"./chroma_{video_id}_{language}"
    embedding = get_embeddings()
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        st.info("Loading existing vector store...")
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding,
            collection_name="youtube_data"
        )
    else:
        if transcript_text is None:
            raise ValueError("Transcript text required to create new store.")
        st.info("Creating new vector store...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.create_documents([transcript_text])
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_dir,
            collection_name="youtube_data"
        )
        
        return vector_store

def create_rag_chain(vector_store):
    llm = get_llm()
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
        llm=llm
    )
    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer only from the provided transcript context.
If the context is insufficient, just say you don't know.
Provide your answer in English.

Context: {context}

Question: {question}
""",
        input_variables=["context", "question"]
    )
    def format_docs(docs):
        return " ".join([doc.page_content for doc in docs])
    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if "chain" not in st.session_state:
    st.session_state.chain = None
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Video Settings")
    video_id = st.text_input("YouTube Video ID", value="J5_-l7WIO_w")
    language = st.selectbox("Transcript Language", ["en", "hi"])
    if st.button("Load Video", type="primary"):
        with st.spinner("Loading transcript and building index..."):
            try:
                transcript_text = load_transcript(video_id, language)
                vector_store = get_vector_store(video_id, language, transcript_text)
                chain = create_rag_chain(vector_store)
                st.session_state.chain = chain
                st.session_state.video_loaded = True
                st.session_state.messages = []
                st.success("✅ Video loaded and indexed successfully!")
            except Exception as e:
                st.error(f"❌ Error: {e}")

if not st.session_state.video_loaded:
    st.info("👈 Please enter a YouTube video ID and click 'Load Video' to start.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask a question about the video:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke({"question": prompt})
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")