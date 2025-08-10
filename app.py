import streamlit as st
from rag_youtube import (
    build_pipeline,
    search_transcript,
    export_text_to_pdf,
    export_text_to_docx,
    export_text_to_csv,
    timestamp_link,
)

st.set_page_config(page_title="YouTube Transcript Q&A (Gemini)", layout="wide")

st.title("YouTube Transcript Question Answering with Gemini")

# Initialize pipeline ONCE
@st.cache_resource
def load_pipeline():
    return build_pipeline()

pipeline = load_pipeline()

main_chain = pipeline["main_chain"]
retriever = pipeline["retriever"]
transcript = pipeline["transcript"]
grouped_chunks = pipeline["grouped_chunks"]

video_id = "qJeaCHQ1k2w"

question = st.text_input("Ask a question about the video:")

language = st.selectbox("Choose answer language:", ["English", "Hindi", "French", "Spanish"])

search_term = st.text_input("Search transcript keywords:")

export_option = st.selectbox("Export transcript as:", ["None", "PDF", "DOCX", "CSV"])

if question:
    try:
        answer = main_chain.invoke({"question": question, "language": language})
        st.markdown(f"### Answer")
        st.markdown(answer)

        st.markdown(f"### Evidence with Timestamps")
        relevant_chunks = retriever.get_relevant_documents(question)
        for chunk in relevant_chunks:
            start_time = chunk.metadata.get("start")
            if start_time:
                ts_url, ts_label = timestamp_link(start_time, video_id)
                st.markdown(f"At [{ts_label}]({ts_url}): {chunk.page_content[:180]}...")
            else:
                st.write(chunk.page_content[:180])
    except Exception as e:
        st.error(f"Error generating answer: {e}")

if search_term:
    snippets = search_transcript(search_term, retriever, video_id)
    st.markdown(f"### Transcript Search Results for '{search_term}'")
    for snippet in snippets:
        st.markdown(snippet, unsafe_allow_html=True)

if export_option != "None":
    if export_option == "PDF":
        export_text_to_pdf(transcript, "transcript.pdf")
        with open("transcript.pdf", "rb") as f:
            st.download_button("Download PDF", f, file_name="transcript.pdf")
    elif export_option == "DOCX":
        export_text_to_docx(transcript, "transcript.docx")
        with open("transcript.docx", "rb") as f:
            st.download_button("Download DOCX", f, file_name="transcript.docx")
    elif export_option == "CSV":
        export_text_to_csv(grouped_chunks, "transcript_chunks.csv")
        with open("transcript_chunks.csv", "rb") as f:
            st.download_button("Download CSV", f, file_name="transcript_chunks.csv")

# You can expand this app with buttons for quiz generation, audio summary, etc.
