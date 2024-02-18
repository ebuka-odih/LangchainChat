from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.callbacks import get_openai_callback
import openai
import os


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() if page.extract_text() else ''
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            # Use the OpenAI client directly for chat completions
            client = openai.OpenAI()  # Make sure to instantiate the correct class
            try:
                # Adjust the model to "gpt-3.5-turbo" and messages as needed
                chat_completion = client.chat.completions.create(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": user_question}]
                )
                # Process and display the response from OpenAI
                if chat_completion.choices:
                    answer = chat_completion.choices[0].message["content"]
                    st.write(answer)
                else:
                    st.error("No response received from OpenAI.")
            except openai.RateLimitError as err:
                st.error("Rate limit exceeded. Please try again later.")
            except openai.OpenAIError as err:
                st.error(f"An error occurred: {str(err)}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == '__main__':
    main()