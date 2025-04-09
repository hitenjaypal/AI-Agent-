from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from main import process_image
import os
import logging
import uvicorn
from fastapi import Body

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def get_favicon():
    favicon_path = os.path.join("static", "favicon.ico")
    if not os.path.exists(favicon_path):
        return Response(status_code=204)  
    return FileResponse(favicon_path)

@app.post("/set_reminder")
async def set_reminder(data: dict):
   
    return {"message": "Reminder saved"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_and_query")
async def upload_and_query(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    try:
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        temp_image_path = os.path.join("static", "temp_upload.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(image_content)

        quiz_type = None
        if "generate" in query.lower() and "quiz" in query.lower():
            if "flashcard" in query.lower():
                quiz_type = "flashcards"
                query += "\n\nPlease extract key concepts and generate exactly 5 flashcards in Q&A format using markdown."
            elif "mcq" in query.lower() or "multiple choice" in query.lower():
                quiz_type = "mcq"
                query += "\n\nPlease generate exactly 5 multiple choice questions with 4 options each. Highlight the correct answer. Use markdown formatting."

        try:
            result = process_image(temp_image_path, query)
            try:
                os.remove(temp_image_path)
            except Exception as cleanup_err:
                logger.warning(f"Cleanup error: {cleanup_err}")

            return JSONResponse(content={
                "llama": result.get("llama", "No response"),
                "llava": result.get("llava", "No response")
            })

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# if __name__ == "__main__":
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
