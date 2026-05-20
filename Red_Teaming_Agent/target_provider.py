import sys
import os
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv()

# Add your target app directory to the path so we can import its modules
target_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Rag_Application'))
sys.path.append(target_app_path)

# --- 1. Initialize your specific target application ---
# Here we initialize the RAG app, mirroring 'persistent_rag_final.ipynb'
from rag_helper import RAGBase
from sqlitesearch import TextSearchIndex

db_path = os.path.join(target_app_path, 'faq.db')
sqlite_index = TextSearchIndex(
    text_fields=['question', 'section', 'answer'],
    keyword_fields=['course'],
    db_path=db_path
)

# openai_client = OpenAI()
target_app = RAGBase(sqlite_index,model='gemma4:31b-cloud')
# --------------------------------------------------------

def call_api(prompt, options, context):
    """
    This is the generic hook required by Promptfoo.
    It takes an adversarial 'prompt' and passes it to your application.
    
    This function is completely agnostic to the type of app. 
    It could be a chat app, a summarizer, an agent, etc.
    """
    try:
        # Pass the adversarial prompt to your target application
        # If your app was a summarizer, you would do: response = summarizer.summarize(prompt)
        response_text = target_app.rag(prompt)
        
        # Return the output in the format Promptfoo expects
        return {"output": response_text}
        
    except Exception as e:
        # Return any errors so Promptfoo can log them
        return {"error": str(e)}
