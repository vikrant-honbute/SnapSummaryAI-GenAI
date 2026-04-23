import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import urllib.error
from requests.exceptions import RequestException, SSLError

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
                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False
                    )
                else:
                    loader = build_web_loader(generic_url)

                docs = loader.load()
            except SSLError:
                if is_youtube:
                    st.warning(
                        "YouTube transcript fetch failed due to an SSL/TLS issue. "
                        "Trying webpage-text fallback..."
                    )
                    try:
                        docs = build_web_loader(generic_url).load()
                    except Exception:
                        st.error(
                            "Could not reach YouTube from this environment due to SSL/network restrictions. "
                            "Please retry later or try a non-YouTube URL."
                        )
                        st.stop()
                else:
                    st.error("SSL/network issue while loading the URL. Please try again.")
                    st.stop()
            except urllib.error.HTTPError:
                st.error(
                    "YouTube returned a 400 Bad Request. "
                    "Try a different video URL (non-private, non-short)."
                )
                st.stop()
            except RequestException:
                st.error("Network error while loading the URL. Please try again.")
                st.stop()
            except Exception as exc:
                st.error(f"Could not load content from this URL. Details: {exc}")
                st.stop()

            if not docs:
                st.error("No content could be extracted from this URL.")
                st.stop()

            # Summarization chain
            chain = load_summarize_chain(
                llm,
                chain_type="stuff",
                prompt=prompt
            )
            try:
                output_summary = chain.run(docs)
                st.write(output_summary)
            except Exception as exc:
                st.error(f"Failed to summarize content. Details: {exc}")
