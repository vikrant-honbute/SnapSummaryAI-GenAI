import os
import html
import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import parse_qs, urlparse

import certifi
import httpx
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
    youtube_debug = st.checkbox("Debug YouTube Reachability", value=False)

generic_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

LANGUAGE_PREFERENCE = ["en", "en-US", "en-orig", "en-GB"]
SUBTITLE_EXT_PREFERENCE = ["json3", "srv3", "vtt", "ttml", "srv1", "srv2"]


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


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()


def debug_youtube_reachability(video_id: str) -> None:
    try:
        response = httpx.get(f"https://www.youtube.com/watch?v={video_id}", timeout=10)
        st.info(f"YouTube reachable: {response.status_code}")
    except Exception as exc:
        st.warning(f"YouTube blocked or unreachable: {exc}")


def parse_json3_subtitle(payload: str) -> str:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return ""

    parts = []
    for event in data.get("events", []):
        for segment in event.get("segs", []):
            piece = segment.get("utf8", "")
            if piece:
                parts.append(piece.replace("\n", " "))

    return normalize_text(" ".join(parts))


def parse_xml_subtitle(payload: str) -> str:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return ""

    chunks = []
    for node in root.iter():
        tag = node.tag.lower() if isinstance(node.tag, str) else ""
        if tag.endswith("text") or tag.endswith("p"):
            text_value = "".join(node.itertext())
            if text_value:
                chunks.append(text_value.replace("\n", " "))

    return normalize_text(" ".join(chunks))


def parse_vtt_subtitle(payload: str) -> str:
    lines = []
    for raw_line in payload.splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.upper().startswith("WEBVTT"):
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        if line.startswith(("NOTE", "STYLE", "REGION")):
            continue
        lines.append(line)

    return normalize_text(" ".join(lines))


def fetch_subtitle_track(track_url: str, ext: str) -> str:
    response = httpx.get(track_url, timeout=15)
    response.raise_for_status()
    payload = response.text

    ext_lower = (ext or "").lower()
    if ext_lower == "json3":
        return parse_json3_subtitle(payload)
    if ext_lower in {"srv3", "ttml", "xml"}:
        return parse_xml_subtitle(payload)
    if ext_lower in {"vtt", "srv1", "srv2"}:
        return parse_vtt_subtitle(payload)

    for parser in (parse_json3_subtitle, parse_xml_subtitle, parse_vtt_subtitle):
        text = parser(payload)
        if text:
            return text

    return ""


def extract_ydlp_subtitles(video_info: dict) -> str:
    ext_rank = {ext: idx for idx, ext in enumerate(SUBTITLE_EXT_PREFERENCE)}
    candidates = []

    subtitle_maps = [
        video_info.get("subtitles", {}) or {},
        video_info.get("automatic_captions", {}) or {},
    ]

    for subtitle_map in subtitle_maps:
        for lang, entries in subtitle_map.items():
            language_rank = (
                LANGUAGE_PREFERENCE.index(lang)
                if lang in LANGUAGE_PREFERENCE
                else len(LANGUAGE_PREFERENCE)
            )

            if not isinstance(entries, list):
                continue

            for entry in entries:
                track_url = entry.get("url")
                ext = (entry.get("ext") or "").lower()
                if not track_url:
                    continue
                candidates.append(
                    (
                        language_rank,
                        ext_rank.get(ext, len(SUBTITLE_EXT_PREFERENCE)),
                        track_url,
                        ext,
                    )
                )

    candidates.sort(key=lambda item: (item[0], item[1]))

    for _, _, track_url, ext in candidates:
        try:
            transcript_text = fetch_subtitle_track(track_url, ext)
            if transcript_text:
                return transcript_text
        except Exception:
            continue

    return ""


def fetch_transcript_with_proxy(video_id: str) -> str:
    proxy_url = os.getenv("YOUTUBE_PROXY_URL", "").strip()
    proxy_username = os.getenv("WEBSHARE_PROXY_USERNAME", "").strip()
    proxy_password = os.getenv("WEBSHARE_PROXY_PASSWORD", "").strip()

    if proxy_username and proxy_password:
        try:
            from youtube_transcript_api.proxies import WebshareProxyConfig

            transcript_api = YouTubeTranscriptApi(
                proxies=WebshareProxyConfig(
                    proxy_username=proxy_username,
                    proxy_password=proxy_password,
                )
            )
            transcript_data = transcript_api.fetch(video_id)
            transcript_text = normalize_text(
                " ".join(
                    item.text if hasattr(item, "text") else item.get("text", "")
                    for item in transcript_data
                )
            )
            if transcript_text:
                return transcript_text
        except Exception:
            pass

    proxy_kwargs = {}
    if proxy_url:
        proxy_kwargs["proxies"] = {"https": proxy_url, "http": proxy_url}

    attempts = [
        {"languages": ["en", "en-US"], **proxy_kwargs},
        {"languages": ["en"], **proxy_kwargs},
        {"languages": ["en", "en-US"]},
    ]

    for kwargs in attempts:
        call_kwargs = dict(kwargs)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, **call_kwargs)
        except TypeError:
            call_kwargs.pop("proxies", None)
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id, **call_kwargs)
            except Exception:
                continue
        except Exception:
            continue

        transcript_text = normalize_text(" ".join(item.get("text", "") for item in transcript))
        if transcript_text:
            return transcript_text

    return ""


def load_youtube_docs(url: str, debug_reachability: bool = False) -> list[Document]:
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not parse YouTube video id from URL.")

    if debug_reachability:
        debug_youtube_reachability(video_id)

    try:
        ydl_options = {
            "quiet": True,
            "skip_download": True,
            "noplaylist": True,
            "socket_timeout": 20,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US"],
            "subtitlesformat": "json3",
        }
        with YoutubeDL(ydl_options) as ydl:
            video_info = ydl.extract_info(url, download=False)
    except Exception as exc:
        transcript_text = fetch_transcript_with_proxy(video_id)
        if transcript_text:
            return [
                Document(
                    page_content=transcript_text,
                    metadata={"source": url, "kind": "transcript_proxy"},
                )
            ]
        raise RuntimeError(f"YouTube fetch failed: {exc}") from exc

    transcript_text = extract_ydlp_subtitles(video_info)
    if transcript_text:
        return [
            Document(
                page_content=transcript_text,
                metadata={"source": url, "kind": "transcript"},
            )
        ]

    transcript_text = fetch_transcript_with_proxy(video_id)
    if transcript_text:
        return [
            Document(
                page_content=transcript_text,
                metadata={"source": url, "kind": "transcript_proxy"},
            )
        ]

    st.warning("No subtitles found. Falling back to video metadata.")

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
                    docs = load_youtube_docs(generic_url, debug_reachability=youtube_debug)
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
