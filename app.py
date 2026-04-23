import os
from urllib.parse import parse_qs, urlparse

import certifi
import streamlit as st
import validators
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from requests.exceptions import RequestException
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# Streamlit App
st.set_page_config(page_title="SnapSummaryAI — YouTube & Web Summarizer", page_icon="🦜")
st.title("🔗📝 SnapSummaryAI — YouTube & Web Summarizer")
st.subheader("Summarize URL")

# Sidebar
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


def is_youtube_url(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def extract_video_id(url: str) -> str | None:
    parsed_url = urlparse(url)
    host = (parsed_url.hostname or "").lower()

    if host in {"youtu.be", "www.youtu.be"}:
        return parsed_url.path.strip("/") or None

    if "youtube.com" in host:
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]

        if parsed_url.path.startswith("/shorts/") or parsed_url.path.startswith("/embed/"):
            path_parts = [part for part in parsed_url.path.split("/") if part]
            if len(path_parts) >= 2:
                return path_parts[1]

    return None


def build_web_loader(url: str) -> UnstructuredURLLoader:
    return UnstructuredURLLoader(
        urls=[url],
        ssl_verify=False,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/116.0.0.0 Safari/537.36"
            )
        },
    )


def load_youtube_docs(url: str) -> list[Document]:
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not parse YouTube video id from URL.")

    transcript = None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US"])
    except Exception:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception:
            st.warning(
                "YouTube transcript fetch failed (often network/IP restriction). "
                "Trying video metadata fallback..."
            )

    if transcript:
        transcript_text = " ".join(item.get("text", "") for item in transcript).strip()
        if transcript_text:
            return [
                Document(
                    page_content=transcript_text,
                    metadata={"source": url, "kind": "transcript"},
                )
            ]

    try:
        with YoutubeDL(
            {
                "quiet": True,
                "skip_download": True,
                "noplaylist": True,
                "socket_timeout": 20,
            }
        ) as ydl:
            video_info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise RuntimeError(f"YouTube transcript and metadata fetch both failed: {exc}") from exc

    title = (video_info.get("title") or "").strip()
    description = (video_info.get("description") or "").strip()
    fallback_text = f"Title: {title}\n\nDescription:\n{description}".strip()

    if fallback_text.replace("Title:", "").replace("Description:", "").strip():
        return [
            Document(
                page_content=fallback_text,
                metadata={"source": url, "kind": "metadata"},
            )
        ]

    raise RuntimeError("No text could be extracted from this YouTube URL.")

if st.button("Summarize the Content from YT or Website"):
    # Validate inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the API key and URL to get started.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It may be a YouTube video URL or website URL.")
    else:
        
        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=groq_api_key
        )

        with st.spinner("Waiting..."):
            # Load data
            is_youtube = is_youtube_url(generic_url)
            try:
                if is_youtube:
                    docs = load_youtube_docs(generic_url)
                else:
                    docs = build_web_loader(generic_url).load()
            except RequestException:
                st.error("Network error while loading the URL. Please try again.")
                st.stop()
            except Exception as exc:
                if is_youtube:
                    st.error(
                        "Could not extract YouTube transcript or metadata in this environment. "
                        f"Details: {exc}"
                    )
                else:
                    st.error(f"Could not load content from this URL. Details: {exc}")
                st.stop()

            if not docs:
                st.error("No content could be extracted from this URL.")
                st.stop()

            combined_text = "\n\n".join(doc.page_content for doc in docs if doc.page_content).strip()
            if not combined_text:
                st.error("Extracted content is empty after processing.")
                st.stop()

            final_prompt = prompt.format(text=combined_text)
            try:
                response = llm.invoke(final_prompt)
                output_summary = response.content if hasattr(response, "content") else str(response)
                st.write(output_summary)
            except Exception as exc:
                st.error(f"Failed to summarize content. Details: {exc}")
