import os
import time
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import SONG_DIR
from src.config import COLLECTION_NAME
from src.config import SUPPORTED_FORMATS
from src.config import VECTOR_SIZE

EMBEDDING_DIM = VECTOR_SIZE
DEFAULT_QUERY_FILE = "query.wav"


def _get_qdrant_client():
    from src.database.qdrant_client import client
    return client


def _get_search_by_vector():
    from src.search.search_engine import search_by_vector
    return search_by_vector


def _get_query_embedding():
    from src.search_song import get_query_embedding
    return get_query_embedding


def _get_record_audio():
    from src.audio.recorder import record_audio
    return record_audio


def _get_index_songs_main():
    from src.index_songs import main as index_songs_main
    return index_songs_main


def count_indexed_songs() -> int:
    files: list[Path] = []
    for ext in SUPPORTED_FORMATS:
        files.extend(SONG_DIR.rglob(f"*{ext}"))
    return len(files)


def count_indexed_vectors() -> int:
    try:
        client = _get_qdrant_client()
        result = client.count(collection_name=COLLECTION_NAME)
        return result.count
    except Exception:
        return 0


def render_sidebar() -> None:
    with st.sidebar:
        st.header("ℹ️ Project Information")
        st.write("**Pipeline:** Demucs → BasicPitch → Melody Filter → Intervals → Embedding")
        st.write("**Embedding:** Deterministic Melody Interval Histogram")
        st.write("**Vector Database:** Qdrant")
        st.write("**Search Method:** Cosine Similarity")
        st.caption("Matches melodic contour only — not singer, lyrics, or genre.")

        st.divider()

        st.subheader("🔧 Index Management")
        if st.button("Rebuild Song Index", use_container_width=True):
            index_songs_main = _get_index_songs_main()
            progress_bar = st.progress(0.0, text="Preparing to index...")
            try:
                def on_progress(completed, total, song_name):
                    if total == 0:
                        return
                    pct = completed / total
                    progress_bar.progress(
                        pct,
                        text=f"Indexing... {completed}/{total} ({pct:.0%}) — {song_name}",
                    )

                index_songs_main(progress_callback=on_progress)
                progress_bar.progress(1.0, text="Indexing complete!")
                st.success("✅ Song index rebuilt successfully!")
            except Exception as e:
                st.error(f"Failed to rebuild index: {e}")


def render_metrics(search_time: float | None = None) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Songs Indexed", count_indexed_songs())
    with col2:
        st.metric("Vectors Indexed", count_indexed_vectors())
    with col3:
        st.metric("Embedding Dimension", EMBEDDING_DIM)
    with col4:
        st.metric("Search Time", f"{search_time:.2f}s" if search_time else "—")


def render_result_card(result: Any) -> None:
    payload = result.payload
    song_name: str = payload.get("title", "Unknown")
    file_path: str = payload.get("path", "")
    similarity_pct = result.score * 100

    with st.container(border=True):
        st.subheader(f"🎵 {song_name}")

        col_info, col_score = st.columns([3, 1])
        with col_info:
            if file_path:
                st.write(f"**File:** `{file_path}`")
        with col_score:
            st.metric("Similarity", f"{similarity_pct:.1f}%")

        st.progress(result.score, text=f"{similarity_pct:.1f}% match")

        if file_path and os.path.exists(file_path):
            with st.expander("▶️ Play Song"):
                st.audio(file_path)


def render_results(results: list[Any]) -> None:
    if not results:
        st.info("No matches found. Try recording again or uploading a different file.")
        return

    st.subheader("🏆 Top Matches")
    for result in results:
        render_result_card(result)


def run_search(audio_path: str, limit: int = 5) -> tuple[list[Any], float]:
    get_query_embedding = _get_query_embedding()
    search_by_vector = _get_search_by_vector()

    with st.spinner("Extracting melody..."):
        st.info("Extracting melody and generating embedding...")
        embedding = get_query_embedding(audio_path)

    st.info("Searching Qdrant...")
    start_time = time.time()
    results = search_by_vector(embedding, top_k=limit)
    search_time = time.time() - start_time

    st.info("Displaying results...")
    return results, search_time


def tab_humming_search() -> None:
    st.markdown("## 🎤 Hum to Search")
    st.write("Click the button below and hum a melody for 10 seconds.")

    col_spacer, col_btn, col_spacer2 = st.columns([2, 1, 2])
    with col_btn:
        record_clicked = st.button("🎙️ Record", use_container_width=True)

    if record_clicked:
        try:
            record_audio = _get_record_audio()
            with st.spinner("Recording 10 seconds of audio..."):
                st.info("Recording... Please hum now!")
                audio_path = record_audio(DEFAULT_QUERY_FILE)
                st.success("Recording saved!")

            results, search_time = run_search(audio_path, limit=5)
            render_results(results)

            st.session_state["last_search_time"] = search_time
        except Exception as e:
            st.error(f"Recording or search failed: {e}")


def tab_file_search() -> None:
    st.markdown("## 📁 Upload Audio File")
    st.write("Upload an audio file (mp3, wav, flac) to search for matching songs.")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["mp3", "wav", "flac"],
    )

    if uploaded_file is not None:
        try:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            st.success(f"Uploaded: {uploaded_file.name}")

            results, search_time = run_search(tmp_path, limit=5)
            render_results(results)

            os.unlink(tmp_path)

            st.session_state["last_search_time"] = search_time
        except Exception as e:
            st.error(f"File search failed: {e}")
            


def main() -> None:
    st.set_page_config(
        page_title="AI Song Finder",
        page_icon="🎵",
        layout="wide",
    )

    render_sidebar()

    st.title("🎵 AI Song Finder")
    st.subheader("Humm a melody and let AI identify your song.")

    search_time = st.session_state.get("last_search_time")
    render_metrics(search_time)

    st.divider()

    tab1, tab2 = st.tabs(["🎤 Search by Humming", "📁 Search Using Audio File"])

    with tab1:
        tab_humming_search()

    with tab2:
        tab_file_search()


if __name__ == "__main__":
    main()
