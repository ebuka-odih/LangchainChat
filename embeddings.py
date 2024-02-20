import fitz  # PyMuPDF
from openai import OpenAI
import numpy as np
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter


# Set your OpenAI API key here
client = OpenAI(api_key='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')

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def generate_embeddings(text, model="text-embedding-ada-002"):
    """Generate embeddings for the given text using OpenAI's API."""
    try:
        # Assuming you have set openai.api_key globally or in an environment variable
        response = client.embeddings.create(
            input=text,
            model=model
        )
        # Extracting and returning the embedding vector
        embedding = response.data[0].embedding
        # response['data'][0]['embedding']
        return embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def process_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return knowledgeBase

def split_text():
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_text(text)
    return documents

# def store_embeddings_in_faiis(embeddings):
#     pdf_path = "test_doc.pdf"
#     text = extract_text_from_pdf(pdf_path)
#     embedding = generate_embeddings(text)
#     db = FAISS.from_texts(documents, OpenAIEmbeddings())

#     return db

# Example usage
pdf_path = "test_doc.pdf"
text = extract_text_from_pdf(pdf_path)
embedding = generate_embeddings(text)

# Store the embedding
vector_store = store_embeddings_in_faiis([embedding])

# Later, you can use vector_store.search(query_embedding, k=1) to find the most similar embeddings
