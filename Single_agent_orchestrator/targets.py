import os
import sys
import json
import http.server
import threading
from dotenv import load_dotenv

# Add Rag_Application directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
target_app_path = os.path.abspath(os.path.join(current_dir, '..', 'Rag_Application'))
if target_app_path not in sys.path:
    sys.path.append(target_app_path)

# Load environment variables
load_dotenv(os.path.join(target_app_path, '.env'))

_target_app = None

def get_rag_app():
    """Lazy initialize and return the target RAG application."""
    global _target_app
    if _target_app is not None:
        return _target_app
        
    from rag_helper import RAGBase
    from sqlitesearch import TextSearchIndex

    db_path = os.path.join(target_app_path, 'faq.db')
    sqlite_index = TextSearchIndex(
        text_fields=['question', 'section', 'answer'],
        keyword_fields=['course'],
        db_path=db_path
    )

    # Use the gemma4 model as configured in RAG application
    _target_app = RAGBase(sqlite_index, model='gemma4:31b-cloud')
    return _target_app


class RAGLocalServer:
    """A simple HTTP server to expose the RAG application as an HTTP endpoint.
    This enables all 4 security frameworks to query the RAG application via REST.
    """
    def __init__(self, port=8000):
        self.port = port
        self.server = None
        self.thread = None
        self.app = get_rag_app()

    def start(self):
        # We need to capture self in the handler, so we define it inside
        class RAGHandler(http.server.BaseHTTPRequestHandler):
            # Capture the RAG application instance
            rag_app = self.app
            
            def do_POST(self):
                if self.path == '/chat':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    
                    try:
                        req_json = json.loads(post_data.decode('utf-8'))
                    except Exception as e:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": f"Invalid JSON: {str(e)}"}).encode('utf-8'))
                        return
                    
                    # Extract the input prompt (checks standard fields across frameworks)
                    prompt = (
                        req_json.get('message') or 
                        req_json.get('prompt') or 
                        req_json.get('input') or 
                        req_json.get('question') or 
                        ""
                    )
                    
                    try:
                        # Query the RAG application
                        response_text = self.rag_app.rag(prompt)
                        # Provide standard keys for different frameworks:
                        # - 'response' for general REST
                        # - 'output' for promptfoo
                        response_data = {
                            'response': response_text,
                            'output': response_text
                        }
                        self.send_response(200)
                    except Exception as e:
                        response_data = {'error': str(e)}
                        self.send_response(500)
                        
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(response_data).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Not Found"}).encode('utf-8'))
                    
            def log_message(self, format, *args):
                # Suppress output logs to keep the console clean
                pass

        self.server = http.server.HTTPServer(('127.0.0.1', self.port), RAGHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"[*] Local RAG Chatbot Server started at http://127.0.0.1:{self.port}/chat")

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("[*] Local RAG Chatbot Server stopped.")
