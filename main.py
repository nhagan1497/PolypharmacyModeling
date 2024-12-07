from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from PIL import Image
import io
from ultralytics import YOLO

from inference import get_pill_properties

app = FastAPI()


def load_model():
    model = YOLO("test_pill_model.pt")
    return model


def get_model():
    # Load the model once and reuse it
    if not hasattr(app.state, "model"):
        app.state.model = load_model()
    return app.state.model


# A simple endpoint to accept an image and API key
@app.post("/process-image")
async def process_image(
        api_key: str,  # Accept the API key as form data
        image: UploadFile,
        model=Depends(get_model)
) -> list[list]:
    valid_api_keys = ["11223344"]
    if api_key not in valid_api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")

    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image")
    image_data = await image.read()
    try:
        # Optional: You can validate the image content here (e.g., using PIL)
        img = Image.open(io.BytesIO(image_data))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    return get_pill_properties(image=img, yolo_model=model)
