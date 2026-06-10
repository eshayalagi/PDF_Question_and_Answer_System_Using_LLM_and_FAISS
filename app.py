import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from fpdf import FPDF

# Set Streamlit page config
st.set_page_config(page_title="🌟 PDF QA SYSTEM", layout="centered")

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("Google_Api_key"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to extract and chunk text from PDFs
def extract_text_from_pdfs(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to create FAISS index using HuggingFace Embeddings
def create_faiss_index(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# Function to query FAISS with the question and get relevant chunks
def query_with_vectorstore(question, vectorstore):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {question}"
    response = model.generate_content(prompt)
    parts = response.candidates[0].content.parts
    return ' '.join(part.text for part in parts)

# Save history with both question and response
def save_history(question, response):
    os.makedirs("history", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    file_name = f"{timestamp}_{uid}.txt"
    file_path = os.path.join("history", file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"You: {question}\n\nGemini: {response}")

# Load history
def load_history():
    history = []
    if not os.path.exists("history"):
        return history
    files = sorted(os.listdir("history"))
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join("history", file), "r", encoding="utf-8") as f:
                content = f.read()
            history.append(content)
    return history

# Function to generate a PDF from chat history
def generate_pdf(chat_history):
    # Create a PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Set font to a Unicode-compatible font (e.g., Arial Unicode MS)
    pdf.set_font("Arial", size=12)  # Arial is fine for most cases, or you can try 'DejaVu Sans'
    
    # Title of the PDF
    pdf.cell(200, 10, txt="Chat History - PDF Analyser", ln=True, align="C")
    pdf.ln(10)

    # Add chat history to the PDF
    for i in range(0, len(chat_history), 2):  # Loop through the chat in pairs (You and Gemini)
        user_message = chat_history[i][1]
        gemini_response = chat_history[i + 1][1] if i + 1 < len(chat_history) else "No response from Gemini"

        # Write the user's message and Gemini's response
        pdf.cell(200, 10, txt=f"You: {user_message}", ln=True)
        pdf.multi_cell(0, 10, txt=f"Gemini: {gemini_response}")
        pdf.ln(5)

    # Save the PDF to a temporary file
    output_path = r"C:\Users\Lenovo\gen4\chat_output.pdf"

    pdf.output(output_path, 'F')

    return output_path

# --- Streamlit App ---

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4e73df;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #2e59d9;
        transform: scale(1.05);
    }
    .stTextInput > div > input {
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    h1, h2, h3 {
        color: #4e73df;
    }
    .user-bubble {
        text-align: right;
        background-color: #d1f5d3;
        color: #1c2b1f;
        padding: 10px;
        margin: 8px 0;
        border-radius: 12px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .bot-bubble {
        text-align: left;
        background-color: #dbe9ff;
        color: #0f1c2e;
        padding: 10px;
        margin: 8px 0;
        border-radius: 12px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌟 PDF QA SYSTEM")
st.header("💬 Ask Questions from Uploaded PDFs")

# Sidebar for file upload
uploaded_files = st.sidebar.file_uploader("📤 Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Process PDFs if uploaded
if uploaded_files:
    text = extract_text_from_pdfs(uploaded_files)
    vectorstore = create_faiss_index(text)
else:
    vectorstore = None

# --- Chat Display Section (Top) ---
if st.session_state.chat:
    st.markdown("### 📜 Chat History")
    for i in range(0, len(st.session_state.chat), 2):  # Loop through the chat in pairs (You and Gemini)
        user_message = st.session_state.chat[i][1]
        gemini_response = st.session_state.chat[i + 1][1] if i + 1 < len(st.session_state.chat) else "No response from Gemini"

        # Display the user's message and the Gemini response together
        st.markdown(
            f"""
            <div class="user-bubble">
                <strong>🧑‍💻 You:</strong> {user_message}
            </div>
            <div class="bot-bubble">
                <strong>🤖 Gemini:</strong> {gemini_response}
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Input Section (Bottom) ---
# Get the user query input
user_query = st.text_input("Ask a question about your PDF:")

# Button to trigger PDF analysis
if st.button("🚀 Ask PDF QA SYSTEM"):
    if not uploaded_files:
        st.warning("⚠ Please upload at least one PDF first!")
    elif not user_query.strip():
        st.warning("⚠ Please enter a question.")
    else:
        with st.spinner("Thinking... 🤖"):
            if vectorstore:
                bot_response = query_with_vectorstore(user_query, vectorstore)
            else:
                bot_response = "⚠ No document uploaded."

        # Save the interaction in session state
        st.session_state.chat.append(("You", user_query))
        st.session_state.chat.append(("Gemini", bot_response))
        save_history(user_query, bot_response)

        # Display the chat interaction
        st.markdown(
            f"""
            <div class="user-bubble">
                <strong>🧑‍💻 You:</strong> {user_query}
            </div>
            <div class="bot-bubble">
                <strong>🤖 Gemini:</strong> {bot_response}
            </div>
            """,
            unsafe_allow_html=True
        )


      

# Show previous extractions
st.markdown("---")
if "show_history" not in st.session_state:
    st.session_state.show_history = False

#if st.button("📚 Show Previous Extractions"):
    #st.session_state.show_history = True

if st.session_state.show_history:
    history = load_history()
    if history:
        st.subheader("📂 Saved Extractions")
        for record in reversed(history):
            st.text_area("📝", value=record, height=200, key="record_note")

    else:
        st.info("No history yet. Upload and ask something!")


# --- PDF Download Section ---
st.markdown("---")
if st.button("📥 Download Chat History as PDF"):
    if st.session_state.chat:
        chat_pdf_path = generate_pdf(st.session_state.chat)
        with open(chat_pdf_path, "rb") as f:
            st.download_button("Download Chat History PDF", f, file_name="chat_history.pdf")
    else:
        st.warning("⚠ No chat history available to download.")
