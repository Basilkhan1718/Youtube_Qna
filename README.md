🎬 YouTube Q&A & Quiz App (RAG with Google Gemini)

This project lets you **ask questions, search, and generate quizzes from a YouTube video's transcript, using Retrieval-Augmented Generation (RAG) with Google Gemini AI.

It:
- Downloads subtitles from YouTube
- Splits them into timestamped chunks
- Embeds them into a FAISS vector database
- Uses a retriever to find relevant transcript segments
- Passes them to Gemini to answer questions only using transcript content
- Supports transcript search, quiz generation, and export to PDF/DOCX/CSV
- Can be run as a Streamlit web app 

---

Features
- Question Answering: Ask anything, get answers based only on the YouTube transcript.
- Transcript Search: Search keywords and jump to exact timestamps.
- Quiz Generator: Create MCQ quizzes from the transcript.
- Export: Save answers or transcripts to PDF, DOCX, or CSV.
- Text-to-Speech: Convert summaries to speech in MP3 format.
- Run Anywhere: Works in Google Colab or locally with Streamlit.

---

Tech Stack
- Python
- LangChain (for RAG pipeline)
- Google Generative AI(`Gemini-2.5-Flash` for Q&A, `Embedding-001` for vector embeddings)
- FAISS (semantic search database)
- yt-dlp + webvtt-py (subtitle download & parsing)
- Streamlit (web UI)
- gTTS, fpdf, python-docx (TTS & export)
