import openai
from dotenv import load_dotenv
import os

# Initialize the OpenAI API key
load_dotenv()
# openai.api_key = st.secrets["OPENAI_API_KEY"]

print(openai.__version__)