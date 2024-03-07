import requests
import cv2
import io
import albumentations as A
from PIL import Image 
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from constants import PARAMS, PRETRAINED_MODEL_PATH
from datasets import FreeParkingPlacesInferenceDataset
from utils import Visualizer
from albumentations.pytorch import ToTensorV2
from test_model import load_model, predict

app = FastAPI(
    title = "Application de détection de places de parking vides",
    description = "Un modèle d'apprentissage automatique pour détecter les places de parking libres à partir d'images satellitaires."
)

@app.get("/", response_class=HTMLResponse)
def serve():
    return """
    <html>
        <head>
            <title></title>
        </head>
        <body>
        <img src="/image">
        <h1>Satellite image</h1>
        </body>
    </html>
    """

def get_image_from_url(url):
    """
    Store an image from a given URL to a specified filename.

    Parameters:
    - url: str. The URL of the image.
    - filename: str. The filename to save the image to.

    Returns:
    - bool: True if the image was saved successfully, False otherwise.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def prediction(image):

    visualizer_worker = Visualizer()

    path_to_saved_model = PRETRAINED_MODEL_PATH
    model = load_model(path_to_saved_model)

    # Setup for testing the model with predefined transformations
    test_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    predictions = predict(model, PARAMS, image, batch_size=16)

    predicted_masks = []
    for predicted_256x256_mask, original_height, original_width in predictions:
        full_sized_mask = cv2.resize(
            predicted_256x256_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        predicted_masks.append(full_sized_mask)

"""
    visualizer_worker.display_image_grid(
        test_image_filenames,
        TEST_IMAGE_DIR,
        TEST_MASK_DIR,
        predicted_masks=predicted_masks,
    )
"""
image = get_image_from_url("https://thumbs.dreamstime.com/b/car-parking-14966337.jpg")
jpeg_img = Image.open(io.BytesIO(image))
print(type(jpeg_img))


@app.get("/image")
async def show_image():
    img = get_image_from_url("https://thumbs.dreamstime.com/b/car-parking-14966337.jpg")
    return Response(content=img, media_type="image/jpeg")



