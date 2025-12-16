import validators, streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import urllib.error

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
            if "youtube.com" in generic_url or "youtu.be" in generic_url:
                loader = YoutubeLoader.from_youtube_url(
                    generic_url,
                    add_video_info=False
                )
            else:
                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/116.0.0.0 Safari/537.36"
                        )
                    },
                )
            try:
                docs = loader.load()
            except urllib.error.HTTPError:
                st.error(
                    "YouTube returned a 400 Bad Request. "
                    "Try a different video URL (non-private, non-short)."
                )
                st.stop()

            # Summarization chain
            chain = load_summarize_chain(
                llm,
                chain_type="stuff",
                prompt=prompt
            )
            output_summary = chain.run(docs)
            st.write(output_summary)
