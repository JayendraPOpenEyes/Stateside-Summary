import os
from dotenv import load_dotenv
load_dotenv()
print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')}")