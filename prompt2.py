import os
import openai
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

# Assuming the OpenAI import and instantiation need correction as per standard usage
# from openai import OpenAI  # This line seems incorrect based on the OpenAI API usage

# Load environment variables
load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() if page.extract_text() else ''
    return text

def process_with_langchain(text, user_question):
    # Placeholder for processing with Langchain
    processed_text = text  # Replace with actual Langchain processing
    return processed_text

def query_openai(processed_text, user_question):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
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
#         return response.choices[0].message['content']
        return response.choices[0].message.content

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return "An error occurred while processing your request."

def main():
    st.set_page_config(page_title="Chat With PDF")
    st.header("Chat With Your PDF 📄")

    pdf_file = st.file_uploader("Upload your PDF", type="pdf")
    if pdf_file is not None:
        text = extract_text_from_pdf(pdf_file)
        processed_text = process_with_langchain(text, "")

        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            response = query_openai(processed_text, user_question)
            st.write("Response from OpenAI:")
            st.write(response)

if __name__ == "__main__":
    main()
