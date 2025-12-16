# SnapSummaryAI – YouTube & Web Summarizer 🚀

SnapSummaryAI is a lightweight Generative AI-powered summarization app that extracts and summarizes content from YouTube videos and web URLs into concise, readable summaries using Groq LLMs and LangChain.  
Built with Streamlit, it is designed for fast inference, clean UX, and easy deployment on Hugging Face Spaces.

---

## ✨ Features

- URL-based summarization  
  - YouTube videos (non-private, non-shorts)  
  - Articles and general web pages  
- Fast inference using Groq-hosted LLMs  
- Context-aware summaries (up to 300 words)  
- Clean Streamlit user interface  
- Secure API key handling (user-provided, not hardcoded)  
- Hugging Face Spaces compatible  

---

## 🛠️ Tech Stack

- Python  
- Streamlit – UI framework  
- LangChain – LLM orchestration  
- Groq LLMs – High-performance inference  
- YoutubeLoader & UnstructuredURLLoader – Content ingestion  

---

## 📂 Project Structure

SnapSummaryAI-GenAI/  
│  
├── app.py                Main Streamlit application  
├── requirements.txt      Project dependencies  
├── README.md             Project documentation  
└── .gitignore            Ignored files (env, venv, cache)  

---

## 🚀 How It Works

1. The user provides:
   - A Groq API key
   - A YouTube or website URL
2. The application:
   - Validates inputs
   - Loads content using the appropriate loader
   - Sends extracted text to a Groq-hosted LLM via LangChain
3. A concise AI-generated summary is displayed in the UI

---

## ▶️ Running Locally

Clone the repository  
git clone https://github.com/vikrant-honbute/SnapSummaryAI-GenAI.git  
cd SnapSummaryAI-GenAI  

Create a virtual environment (recommended)  
python -m venv venv  
source venv/bin/activate   (macOS/Linux)  
venv\Scripts\activate      (Windows)  

Install dependencies  
pip install -r requirements.txt  

Run the application  
streamlit run app.py  

---

## 🔑 API Key Handling

- The Groq API key is entered via the Streamlit sidebar  
- The key is not stored, not logged, and not committed  
- Safe for local use, public demos, and Hugging Face Spaces  

---

## 🌐 Deployment (Hugging Face Spaces)

SnapSummaryAI is fully compatible with Hugging Face Spaces.

Deployment steps:
1. Create a new Space and select Streamlit
2. Connect this GitHub repository
3. Ensure app.py and requirements.txt are present
4. Launch the Space

Users can securely provide their own Groq API key from the UI.

---

## ⚠️ Limitations

- Does not support:
  - Private or restricted YouTube videos
  - Extremely large web pages without preprocessing
- Summary length is currently fixed

---

## 📌 Future Improvements

- Streaming summaries (token-by-token)
- Chapter-wise summaries for long videos
- Multi-language summarization
- Chat-style conversational memory
- UI theming and dark mode

---

## 👨‍💻 Author

Vikrant Honbute  
AI / ML Engineer | GenAI & LLM Applications  
GitHub: https://github.com/vikrant-honbute  

---

## 📜 License

This project is open-source and intended for educational and demonstration purposes.
