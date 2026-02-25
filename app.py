import streamlit as st
import os
from src.ingestion import IngestionSystem
from src.retrieval import RetrievalSystem
from src.generation import GeminiTutor

st.set_page_config(page_title="Scholar RAG", page_icon="🎓", layout="wide")
st.title("🎓 Scholar RAG: The AI Tutor")

# --- STATE ---
if "retriever" not in st.session_state: st.session_state.retriever = None
if "tutor" not in st.session_state: st.session_state.tutor = GeminiTutor()
if "messages" not in st.session_state: st.session_state.messages = []

# --- SIDEBAR ---
# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    if st.button("Process Textbook") and uploaded_file:
        with st.spinner("Processing..."):
            temp_path = "temp.pdf"
            with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            ingestor = IngestionSystem()
            chunks = ingestor.load_pdf(temp_path)
            
            retriever = RetrievalSystem()
            retriever.embed_documents(chunks)
            
            st.session_state.retriever = retriever
            os.remove(temp_path)
            st.success("Knowledge Base Ready! ✅")
            
    st.divider() # Adds a nice visual line
    
    # --- NEW FEATURE: STUDY GUIDE ---
    st.header("2. 📝 Study Tools")
    if st.button("Generate Study Guide"):
        if len(st.session_state.messages) == 0:
            st.warning("Chat with your document first to build a guide!")
        else:
            with st.spinner("Compiling your notes..."):
                # Call our new method
                guide_content = st.session_state.tutor.generate_study_guide(st.session_state.messages)
                st.session_state.study_guide = guide_content
                st.success("Study Guide Generated!")

    # If a guide exists in memory, show the download button
    # If a guide exists in memory, show the download buttons
    if "study_guide" in st.session_state:
        # Create two columns so the buttons sit side-by-side
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📥 as .md",
                data=st.session_state.study_guide,
                file_name="Study_Guide.md",
                mime="text/markdown",
                use_container_width=True # Makes the button fill the column
            )
            
        with col2:
            st.download_button(
                label="📄 as .txt",
                data=st.session_state.study_guide,
                file_name="Study_Guide.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Give them a sneak peek
        with st.expander("Preview Study Guide"):
            st.markdown(st.session_state.study_guide)

# --- CHAT ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are sources attached to this message, show them
        if "sources" in message:
            with st.expander("📚 Sources & Page Numbers"):
                for source in message["sources"]:
                    st.markdown(f"**Page {source['page']}**: {source['text'][:150]}...")

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.retriever:
            # Note: We removed the 'st.spinner' because streaming starts instantly!
            
            # 1. Retrieve Context
            retrieved_items = st.session_state.retriever.retrieve(prompt, top_k=5)
            context_text = "\n\n".join([item["text"] for item in retrieved_items])
            
            # 2. Ask Gemini (This now returns a stream generator)
            stream = st.session_state.tutor.ask(prompt, context_text, st.session_state.messages)
            
            # 3. Stream to UI (st.write_stream automatically types it out and returns the full text)
            full_answer = st.write_stream(stream)
            
            # 4. Display Citations
            unique_pages = sorted(list(set([item["page"] for item in retrieved_items])))
            st.caption(f"Sources found on pages: {', '.join(map(str, unique_pages))}")
            
            with st.expander("🔍 View Source Excerpts"):
                for item in retrieved_items:
                    st.markdown(f"**Page {item['page']}**")
                    st.info(item['text'][:300] + "...")
            
            # Save to history (we save 'full_answer' so memory works perfectly)
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_answer,
                "sources": retrieved_items 
            })
        else:
            st.error("⚠️ Please upload a PDF first.")