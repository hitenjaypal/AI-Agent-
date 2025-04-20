import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env file")

def process_image(image_path, query):
    try:
        abs_path = os.path.abspath(image_path)
        logger.info(f"Processing image: {abs_path}")

        # (Optional) Load and verify image if you still want
        with open(abs_path, "rb") as f:
            image_content = f.read()

        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image: {str(e)}")
            return {"error": f"Invalid image: {str(e)}"}

        # ONLY send text query (no image) to GROQ
        messages = [
            {"role": "user", "content": query}
        ]

        def make_api_request(model):
            try:
                response = requests.post(
                    GROQ_API_URL,
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 1000
                    },
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                return None

        llama_response = make_api_request("llama-3.3-70b-versatile")   # Correct model
        llava_response = make_api_request("llama-3.3-8b-versatile")    # Correct model

        responses = {}
        for model, response in [("llama", llama_response), ("llava", llava_response)]:
            if response and response.status_code == 200:
                result = response.json()
                responses[model] = result["choices"][0]["message"]["content"]
            else:
                logger.error(f"API error ({model}) or No response.")
                responses[model] = "API error or No response."

        return responses

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {"error": str(e)}

