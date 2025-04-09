import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging

TRANSLATE_API = "https://libretranslate.de"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in .env file")

def detect_language(text):
    try:
        response = requests.post(
            f"{TRANSLATE_API}/detect",
            headers={"Content-Type": "application/json"},
            json={"q": text}
        )
        detections = response.json()
        if detections and isinstance(detections, list):
            return detections[0]['language']
        return "en"
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")
        return "en"

def translate_text(text, source_lang="auto", target_lang="en"):
    try:
        response = requests.post(
            f"{TRANSLATE_API}/translate",
            headers={"Content-Type": "application/json"},
            json={
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text"
            }
        )
        result = response.json()
        return result["translatedText"]
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return text  

def process_image(image_path, query, history=None):
    try:
        original_query_lang = detect_language(query)
        logger.info(f"Detected user language: {original_query_lang}")
        
        translated_query = translate_text(query, source_lang=original_query_lang, target_lang="en")

        abs_path = os.path.abspath(image_path)
        logger.info(f"Processing image: {abs_path}")
        
        with open(abs_path, "rb") as f:
            image_content = f.read()
            encoded_image = base64.b64encode(image_content).decode("utf-8")

        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image: {str(e)}")
            return {"error": f"Invalid image: {str(e)}"}

        messages = history if history else []
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": translated_query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        })

        def make_api_request(model):
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
            return response

        llama_response = make_api_request("llama-3.2-11b-vision-preview")
        llava_response = make_api_request("llama-3.2-90b-vision-preview")

        responses = {}
        for model, response in [("llama", llama_response), ("llava", llava_response)]:
            if response.status_code == 200:
                result = response.json()
                content_en = result["choices"][0]["message"]["content"]

                logger.info(f"Translating {model}'s response back to {original_query_lang}")
                content_translated = translate_text(content_en, source_lang="en", target_lang=original_query_lang)
                responses[model] = content_translated

                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": content_en}]
                })
            else:
                logger.error(f"API error ({model}): {response.status_code}")
                responses[model] = f"API error: {response.status_code}"

        return {
            "llama": responses["llama"],
            "llava": responses["llava"],
            "history": messages
        }

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {"error": str(e)}
