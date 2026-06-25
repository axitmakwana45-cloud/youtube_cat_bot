🎥 YouTube Video Chat with RAG
Ask questions about any YouTube video in real time, using a Retrieval-Augmented Generation (RAG) pipeline built with LangChain.

https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
https://img.shields.io/badge/Python-3.9%252B-blue
https://img.shields.io/badge/LangChain-0.2%252B-green
https://img.shields.io/badge/License-MIT-yellow

🚀 Features
📥 Load any YouTube video – just paste the video ID and choose the transcript language.

🧠 RAG-powered Q&A – ask questions and get answers grounded only in the video’s transcript.

💾 Persistent vector store – once indexed, the video is cached locally; no need to re‑process on every run.

⚡ Real‑time chat interface – interactive chat with conversation history.

🔍 Multi‑query retrieval – uses MultiQueryRetriever to improve relevance.

🌐 Multilingual support – works with both English and Hindi transcripts (and many other languages supported by YouTube).

🛠️ Technologies Used
Component	Library/Tool
App Framework	Streamlit
RAG Pipeline	LangChain (classic)
Vector Store	Chroma (persistent)
Embeddings	MistralAI (mistral-embed)
LLM	Groq (llama-4-scout-17b-16e-instruct)
Transcript Fetch	youtube-transcript-api
Text Splitting	RecursiveCharacterTextSplitter

📋 Prerequisites
Python 3.9 or higher

API keys for:

Mistral AI – for embeddings

Groq – for the LLM

🧠 How It Works (High‑Level)
Indexing

The YouTube transcript is fetched and split into overlapping chunks.

Each chunk is embedded with MistralAI and stored in a Chroma vector store.

Retrieval

When a question is asked, the MultiQueryRetriever generates multiple rephrased versions of the query to improve recall.

The retriever performs an MMR (Maximum Marginal Relevance) search to find the most relevant and diverse chunks.

Generation

Retrieved chunks and the original question are passed to a prompt template.

The LLM (Groq’s LLaMA‑4) generates an answer strictly from the provided context.

Chain

The entire pipeline is orchestrated using LangChain’s RunnableParallel, ensuring clean and modular execution.

🧪 Example Questions
“What is the main topic of this video?”

“Can you summarise the video in 5 bullet points?”

“Was nuclear fusion discussed? If yes, what were the key points?”

“Who is Demis?” (if mentioned in the transcript)
