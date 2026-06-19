# 🚀 Deployment Guide

This project is built using **Streamlit + Groq + FAISS + Sentence Transformers**.

---

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/AICTE-BATCH1-AI-PDF-Study-Assistant.git
cd AICTE-BATCH1-AI-PDF-Study-Assistant
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Groq API Key

For security reasons, the API key is **not included** in this repository.

Replace the following code inside `app.py`:

```python
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)
```

with:

```python
client = Groq(
    api_key="YOUR_GROQ_API_KEY"
)
```

Get your API key from:

https://console.groq.com/keys

---

## 5️⃣ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser automatically.

---

# Features

- 💬 Chat with PDF
- 📝 Smart Notes Generation
- 🃏 Flashcards
- ❓ Quiz Generator
- 📄 Download Notes as PDF

---

# Important Note

This repository does **not** include:

- Groq API keys
- Secret files
- Environment files

Users must provide their own API key before running the project.

---

## Tech Stack

- Streamlit
- Groq API
- LangChain
- FAISS
- Sentence Transformers
- PyPDF2
- FPDF

---
