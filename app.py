from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import openai
from openai import OpenAI
import os

# Initialize the OpenAI API key
load_dotenv()
# openai.api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a function to process text with Langchain
def process_with_langchain(text, user_question):
    # Placeholder for processing text with Langchain
    processed_text = text  # Replace this with actual Langchain processing if necessary
    return processed_text

# Define a function to query OpenAI
def query_openai(processed_text, user_question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"The following is a summary of the document: {processed_text[:1024]}..."},
                {"role": "user", "content": user_question}
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message['content']
    except openai.error.RateLimitError as err:
        return f"OpenAI error: {str(err)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Define the main function for Streamlit app
def main():
    st.set_page_config(page_title="Chat With PDF")
    st.header("Chat With PDF📄")
    
    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''
        
        # Process text with Langchain
        processed_text = process_with_langchain(text, "")

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(processed_text)
        
        # Create embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # Show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            response = query_openai(processed_text, user_question)
            st.write(response)

if __name__ == '__main__':
    main()
