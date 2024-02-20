import streamlit as st
import os
from PIL import Image
import fitz  # PyMuPDF
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

# Initialize global variables
chat_history = []
chain = None
N = 0

# Function to set the OpenAI API key
def set_apikey(api_key):
    os.environ['OPENAI_API_KEY'] = api_key
    st.success("OpenAI API key is set.")

# Function to process the PDF file and create a conversation chain
def process_file(file):
    if 'OPENAI_API_KEY' not in os.environ:
        st.error('Upload your OpenAI API key')
        return None

    loader = PyPDFLoader(file.name)
    documents = loader.load()

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=1024)
    pdfsearch = Chroma.from_documents(documents, embeddings)

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.3), 
                                                  retriever=pdfsearch.as_retriever(search_kwargs={"k": 1}),
                                                  return_source_documents=True)
    return chain

# Main application
def main():
    st.title("PDF Conversational Interface")

    # API Key Input
    api_key = st.text_input("Enter OpenAI API key", type="password")
    if api_key:
        set_apikey(api_key)

    # PDF Upload
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None:
        global chain
        with st.spinner('Processing PDF...'):
            chain = process_file(uploaded_file)

    # Chat Interface Simulation
    user_query = st.text_input("Enter your query", key="query_input")
    submit_query = st.button("Submit")

    if submit_query and user_query and chain:
        global chat_history, N
        result = chain({"question": user_query, 'chat_history': chat_history}, return_only_outputs=True)
        chat_history += [(user_query, result["answer"])]
        N = list(result['source_documents'][0])[1][1]['page']

        # Display chat history
        for q, a in chat_history:
            st.text_area("Q:", value=q, height=50, disabled=True)
            st.text_area("A:", value=a, height=100, disabled=True)

        # Render and display PDF page referred in the last answer
        if uploaded_file:
            doc = fitz.open(uploaded_file.name)
            page = doc[N]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(image, caption=f"Page {N+1}")

if __name__ == "__main__":
    main()
