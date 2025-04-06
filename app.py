from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from main import process_image
import os
import logging
import uvicorn
from fastapi import Body


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Create required directories
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Favicon route
@app.get("/favicon.ico")
async def get_favicon():
    favicon_path = os.path.join("static", "favicon.ico")
    if not os.path.exists(favicon_path):
        return Response(status_code=204)  # No content if no favicon
    return FileResponse(favicon_path)

# Main route
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Image processing route
@app.post("/upload_and_query")
async def upload_and_query(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    try:
        # Save the uploaded file temporarily
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Create a temporary file
        temp_image_path = os.path.join("static", "temp_upload.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(image_content)
        
        try:
            # Process the image using main.py function
            result = process_image(temp_image_path, query)
            
            # Clean up the temporary file
            try:
                os.remove(temp_image_path)
            except:
                pass
            
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
