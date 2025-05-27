import os
import gradio as gr
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS  # ✅ Fixed import
from langgraph.graph import StateGraph, START
from typing_extensions import TypedDict, List
import tempfile
import shutil
import time

# ✅ Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise EnvironmentError("❌ GROQ_API_KEY not found in .env file. Please define it.")

# ✅ Init LLaMA-3 via Groq
try:
    llm = init_chat_model("llama3-8b-8192", model_provider="groq")
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    raise

# ✅ Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)

# ✅ Prompt template
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use twenty sentences maximum and keep the answer as detailed as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

# ✅ State type
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# ✅ Global variables
vector_store = None
embeddings = None

# ✅ Initialize embeddings once at startup
def init_embeddings():
    global embeddings
    try:
        print("🔄 Initializing embeddings model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        print("✅ Embeddings model initialized successfully")
        return True  # ✅ Fixed return value
    except Exception as e:
        print(f"❌ Error initializing embeddings: {e}")
        return False

# ✅ PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ''
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ''
            text += page_text
            if i % 10 == 0:  # Progress indicator
                print(f"📄 Processed {i+1}/{len(reader.pages)} pages")
        return text
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        raise

# ✅ Process PDF -> Vector store with timeout
def process_pdf(pdf_file):
    global vector_store, embeddings
    
    if pdf_file is None:
        return "❌ No PDF file uploaded."
    
    # Initialize embeddings if not already done
    if embeddings is None:
        if not init_embeddings():
            return "❌ Failed to initialize embeddings model. Check your internet connection."
    
    print("🔄 Starting PDF processing...")
    
    # Extract text
    text = extract_text_from_pdf(pdf_file.name)
    if not text.strip():
        return "❌ No text could be extracted from the PDF."
    
    print(f"📄 Extracted {len(text)} characters from PDF")
    
    # Split text
    print("🔄 Splitting text into chunks...")
    all_splits = text_splitter.split_text(text)
    print(f"📝 Created {len(all_splits)} text chunks")
    
    if len(all_splits) == 0:
        return "❌ No text chunks created from the PDF."
    
    # Create documents
    documents = [
        Document(page_content=split, metadata={"chunk_id": i}) 
        for i, split in enumerate(all_splits)
    ]
    
    # Create vector store
    print("🔄 Creating FAISS vector store...")
    start_time = time.time()
    
    try:
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        print("✅ FAISS vector store created successfully")
        print("🎉 PDF processing completed!")
        print("🕒 Time taken:", round(time.time() - start_time, 2), "seconds")
        
        return f"✅ PDF processed successfully! Created {len(all_splits)} text chunks and indexed them for search."
    
    except Exception as e:
        print(f"❌ FAISS error: {e}")
        return "❌ Failed to create FAISS vector store."

# ✅ Retriever + Generator functions
def retrieve(state: State):
    if vector_store is None:
        return {"context": []}
    try:
        retrieved_docs = vector_store.similarity_search(state["question"], k=4)
        return {"context": retrieved_docs}
    except Exception as e:
        print(f"❌ Error in retrieval: {e}")
        return {"context": []}

def generate(state: State):
    if not state["context"]:
        return {"answer": "❌ No context available. Please upload and process a PDF first."}
    
    try:
        context_str = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke({"question": state["question"], "context": context_str})
        response = llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        error_msg = f"❌ Error generating answer: {str(e)}"
        print(error_msg)
        return {"answer": error_msg}

# ✅ LangGraph flow
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# ✅ Handler for question
def ask_question(user_question):
    if not user_question.strip():
        return "❌ Please enter a question."
    
    if vector_store is None:
        return "❌ No PDF processed yet. Please upload and process a PDF first."
    
    try:
        print(f"🔄 Processing question: {user_question}")
        result = graph.invoke({"question": user_question})
        return result["answer"]
    except Exception as e:
        error_msg = f"❌ Error generating answer: {str(e)}"
        print(error_msg)
        return error_msg

# ✅ Simplified Gradio Interface to avoid schema issues
def create_interface():
    def combined_process_and_ask(pdf_file, question):
        """Combined function to process PDF and ask question"""
        if pdf_file is None:
            return "❌ No PDF file uploaded.", ""
        
        # Process PDF
        status = process_pdf(pdf_file)
        
        # If processing successful and question provided, ask question
        if "✅" in status and question and question.strip():
            answer = ask_question(question)
            return status, answer
        
        return status, ""
    
    # Create interface with simpler structure
    interface = gr.Interface(
        fn=combined_process_and_ask,
        inputs=[
            gr.File(label="📄 Upload PDF", file_types=[".pdf"]),
            gr.Textbox(label="❓ Ask Question (optional)", placeholder="Enter your question here...")
        ],
        outputs=[
            gr.Textbox(label="📊 Processing Status", lines=3),
            gr.Textbox(label="💡 Answer", lines=10)
        ],
        title="📘 PDF Q&A Assistant",
        description="Upload a PDF document and optionally ask a question about its content!",
        allow_flagging="never"
    )
    
    return interface

# Alternative: Minimal Blocks interface
def create_simple_blocks():
    with gr.Blocks(title="PDF Q&A Assistant") as demo:
        gr.Markdown("## 📘 PDF Q&A Assistant")
        
        pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
        question_input = gr.Textbox(label="❓ Question", placeholder="Ask about the PDF...")
        
        process_btn = gr.Button("Process & Ask")
        
        status_output = gr.Textbox(label="Status", lines=2)
        answer_output = gr.Textbox(label="Answer", lines=8)
        
        def process_and_ask(pdf_file, question):
            if pdf_file is None:
                return "❌ No PDF uploaded", ""
            
            status = process_pdf(pdf_file)
            if "✅" in status and question.strip():
                answer = ask_question(question)
                return status, answer
            return status, ""
        
        process_btn.click(
            fn=process_and_ask,
            inputs=[pdf_input, question_input],
            outputs=[status_output, answer_output]
        )
    
    return demo

# ✅ Initialize embeddings at startup
print("🚀 Starting application...")
if init_embeddings():
    print("✅ Ready to process PDFs!")
else:
    print("⚠️ Warning: Embeddings not initialized. Will try again when processing first PDF.")

if __name__ == "__main__":
    # Try the simpler interface first
    try:
        demo = create_simple_blocks()
        demo.launch(
            server_name="127.0.0.1",  # ✅ Changed for Docker compatibility
            server_port=7860,
            share=False,  # ✅ Disabled share for Docker
            debug=False   # ✅ Disabled debug to reduce complexity
        )
    except Exception as e:
        print(f"❌ Error launching Gradio interface: {e}")
        # Fallback to even simpler interface
        try:
            demo = create_interface()
            demo.launch(
                server_name="127.0.0.1",
                server_port=7860,
                share=False
            )
        except Exception as e2:
            print(f"❌ Fallback interface also failed: {e2}")