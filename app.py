import streamlit as st
import PyPDF2
import json
import re
from groq import Groq
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS


st.set_page_config(
    page_title="AI PDF Study Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Study Assistant")
st.markdown(
    """
Upload a PDF and generate:

- 💬 Chat with PDF
- 📝 Smart Notes
- 🃏 Flashcards
- ❓ Quiz
"""
)

# If you are running this project locally,
# replace the below code with:

client = Groq(
     api_key="YOUR_GROQ_API_KEY"
 )

#client = Groq(
#    api_key=st.secrets["GROQ_API_KEY"]
#)


class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self):
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(text)
        return embedding.tolist()


embedding_model = SentenceTransformerEmbeddings()


def extract_text_from_pdf(pdf_file):

    reader = PyPDF2.PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=1000, overlap=200):

    if not text.strip():
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_vector_store(chunks):

    if not chunks:
        return None

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model
    )

    return vector_store


def retrieve_context(
        vector_store,
        query,
        k=8):

    docs = vector_store.similarity_search(
        query,
        k=k
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    return context


def clean_json(text):

    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    start = min(
        [i for i in [text.find("["), text.find("{")] if i != -1],
        default=0
    )

    text = text[start:]

    return text


def clean_notes(text):

    text = re.sub(r"#+", "", text)
    text = text.replace("**", "")
    text = text.replace("```", "")

    return text.strip()


def export_notes_pdf(notes_text):

    pdf = FPDF()

    pdf.add_page()

    pdf.set_auto_page_break(True, margin=15)

    pdf.set_font("Arial", size=12)

    for line in notes_text.split("\n"):

        pdf.multi_cell(0, 8, line)

    filename = "notes.pdf"

    pdf.output(filename)

    return filename

def generate_with_groq(prompt):

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
You are an expert AI study assistant.

Rules:
- Give clear answers.
- Use simple language.
- Never return markdown code blocks.
- Never surround JSON with ```json.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1
    )

    return response.choices[0].message.content

def chat_with_pdf(vector_store, user_question):

    context = retrieve_context(
        vector_store,
        user_question
    )

    prompt = f"""
Answer the question only using the context below.

CONTEXT:
{context}

QUESTION:
{user_question}
"""

    return generate_with_groq(prompt)


def generate_notes(vector_store):

    context = retrieve_context(
        vector_store,
        "Generate detailed study notes"
    )

    prompt = f"""
Create neat study notes.

Rules:
- Use simple headings.
- Use bullet points.
- Avoid ### and ** symbols.
- Make notes exam-oriented.
- Keep formatting clean.

CONTENT:
{context}
"""

    notes = generate_with_groq(prompt)

    return clean_notes(notes)


def generate_flashcards(vector_store):

    context = retrieve_context(
        vector_store,
        "important concepts and definitions"
    )

    prompt = f"""
Create flashcards.

Return ONLY JSON.

Example:

[
 {{
   "question":"What is Python?",
   "answer":"Python is a programming language."
 }}
]

CONTENT:
{context}
"""

    result = generate_with_groq(prompt)

    try:

        result = clean_json(result)

        flashcards = json.loads(result)

        return flashcards

    except:

        return []


def generate_quiz(vector_store):

    context = retrieve_context(
        vector_store,
        "important concepts"
    )

    prompt = f"""
Create exactly 5 NEW multiple-choice questions.

Rules:
- Questions should be different every time.
- Avoid repeating previous questions.
- Randomly choose concepts from the content.
- Provide 4 options for each question.
- Return ONLY JSON.

Example:

[
 {{
   "question":"Python is?",
   "options":["Language","Animal","Fruit","City"],
   "correct_answer":"Language"
 }}
]

CONTENT:
{context}
"""

    result = generate_with_groq(prompt)

    try:

        result = clean_json(result)

        quiz = json.loads(result)

        return quiz

    except:

        return []


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "notes" not in st.session_state:
    st.session_state.notes = None

if "flashcards" not in st.session_state:
    st.session_state.flashcards = None

if "quiz" not in st.session_state:
    st.session_state.quiz = None


st.sidebar.header("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)


if uploaded_file:

    with st.spinner("Processing PDF..."):

        text = extract_text_from_pdf(uploaded_file)
        st.write("Text length:", len(text))

        chunks = split_text(text)
        st.write("Number of chunks:", len(chunks))

        if len(chunks) > 0:

            vector_store = create_vector_store(chunks)
            st.session_state.vector_store = vector_store

        else:

            st.error(
                "No text could be extracted from this PDF. "
                "Please upload a text-based PDF."
            )

    st.sidebar.success("PDF uploaded successfully!")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💬 Chatbot",
        "📝 Notes",
        "🃏 Flashcards",
        "❓ Quiz",
    ]
)


with tab1:

    st.subheader("Chat with PDF")

    if st.session_state.vector_store:

        question = st.text_input(
            "Ask anything from your PDF"
        )

        send = st.button("Send")

        if "last_question" not in st.session_state:
            st.session_state.last_question = ""

        if question and (
                send or
                question != st.session_state.last_question):

            st.session_state.last_question = question

            with st.spinner("Thinking..."):

                answer = chat_with_pdf(
                    st.session_state.vector_store,
                    question
                )

                st.session_state.chat_history.append(
                    (question, answer)
                )

        for q, a in reversed(st.session_state.chat_history):

            st.markdown(f"**You:** {q}")
            st.info(a)

    else:

        st.warning("Please upload a PDF first.")


with tab2:

    st.subheader("Generate Study Notes")

    if st.session_state.vector_store:

        if st.button("Generate Notes"):

            with st.spinner("Generating Notes..."):

                st.session_state.notes = generate_notes(
                    st.session_state.vector_store
                )

        if st.session_state.notes:

            st.markdown(st.session_state.notes)

            pdf_file = export_notes_pdf(
                st.session_state.notes
            )

            with open(pdf_file, "rb") as file:

                st.download_button(
                    "⬇ Download Notes PDF",
                    file,
                    file_name="notes.pdf"
                )

    else:

        st.warning("Please upload a PDF first.")


with tab3:

    st.subheader("Flashcards")

    if st.session_state.vector_store:

        if st.button("Generate Flashcards"):

            with st.spinner("Generating Flashcards..."):

                st.session_state.flashcards = generate_flashcards(
                    st.session_state.vector_store
                )

        if st.session_state.flashcards:
            flashcard_text = ""
            for i, card in enumerate(
                st.session_state.flashcards):
                st.markdown(f"### Card {i+1}")
                st.info(card.get("question", ""))

                st.success(card.get("answer", ""))

                flashcard_text += (
                f"Card {i+1}\n"
                f"Q: {card.get('question','')}\n"
                f"A: {card.get('answer','')}\n\n"
            )

            st.download_button(
                "⬇ Download Flashcards",
                flashcard_text,
                file_name="flashcards.txt"
           )

    else:

        st.warning("Please upload a PDF first.")

with tab4:

    st.subheader("Quiz")

    if st.session_state.vector_store:

        if st.button("Generate Quiz"):
            st.session_state.quiz = None

            with st.spinner("Generating Quiz..."):

                st.session_state.quiz = generate_quiz(
                    st.session_state.vector_store
                )

        if st.session_state.quiz:

            score = 0

            for i, q in enumerate(st.session_state.quiz):

                st.markdown(
                    f"### Question {i+1}"
                )

                st.write(q["question"])

                answer = st.radio(
                    "",
                    q["options"],
                    index=None,
                    key=f"quiz_{i}"
                )

                if answer == q["correct_answer"]:
                    score += 1

            if st.button("Show Score"):
                st.success(
                    f"Score: {score}/{len(st.session_state.quiz)}"
                )
                for i, q in enumerate(st.session_state.quiz):
                    selected = st.session_state.get(f"quiz_{i}")
                    st.markdown(f"### Question {i+1}")
                    st.write("Your Answer:", selected)
                    st.write(
                        "Correct Answer:",
                        q["correct_answer"]
                        )
                    if selected == q["correct_answer"]:
                        st.success("Correct ✅")
                    else:
                        st.error("Wrong ❌")

    else:

        st.warning("Please upload a PDF first.")
        
st.markdown("---")
