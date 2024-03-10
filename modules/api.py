import os
import glob
import cv2
import matplotlib.pyplot as plt
import requests
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from constants import PARAMS, PRETRAINED_MODEL_PATH
from datasets import FreeParkingPlacesInferenceDataset
from utils import Visualizer
from fastapi.staticfiles import StaticFiles
from test_model import load_model, predict



def clean_folder(folder_path):
    """
    Deletes all files in the specified folder (here used for the data/API folder).

    Args:
    - folder_path (str): The path to the folder from which all files will be deleted.
    """
    # List all files in the folder
    files = glob.glob(os.path.join(folder_path, '*'))
    
    # Remove each file.
    for file_path in files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error occurred while deleting file {file_path}: {e}")

def prediction(filename):
    """
    Deletes all files in the specified folder (here used for the data/API folder).

    Args:
    - filename (str): The path to the folder which contains the image to predict the places from.
    """

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
    test_dataset = FreeParkingPlacesInferenceDataset(
        filename, transform=test_transform
    )

    predictions = predict(model, PARAMS, test_dataset, batch_size=16)

    predicted_masks = []
    for predicted_256x256_mask, original_height, original_width in predictions:
        full_sized_mask = cv2.resize(
            predicted_256x256_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        predicted_masks.append(full_sized_mask)
    
    return predicted_masks


def plot_prediction(pred, save_path):
    """
    Plot the prediction array and save it as an image.

    Args:
    - pred: The prediction array to plot.
    - save_path: Path where the plot image will be saved.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(pred[0])
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Creation of the FastAPI application
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def serve():
    return """
    <html>
        <head>
            <title>Image Display</title>
        </head>
        <body>
            <h1>Enter the URL of the image</h1>
            <form action="/fetch-and-display-image" method="post">
                <input type="text" name="image_url" placeholder="Enter Image URL here">
                <button type="submit">Show Prediction</button>
            </form>
        </body>
    </html>
    """


@app.post("/fetch-and-display-image")
async def fetch_and_display_image(image_url: str = Form(...)):
    """
    Fetch an image from the provided URL, save it as a PNG image in a specific folder,
    calls the model to predict and save the prediction in order to display it.
    """
    #folder where you want to save the image
    save_folder = Path("../data/API")
    app.mount("/static", StaticFiles(directory="../data/API"), name="static")
    save_folder.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist
    clean_folder(save_folder)

    # Extract the image name from the URL
    image_name = Path(image_url).name
    save_path = save_folder / f"{image_name}.png"

    try:
        # Fetch the image using requests
        response = requests.get(image_url)
        response.raise_for_status()

        with open(save_path, 'wb') as file:
            file.write(response.content)

        # Make the prediction
        pred = prediction(save_folder)

        plot_path = save_folder / "prediction_plot.png"
        plot_prediction(pred, plot_path)

        html_content = f"""
        <html>
            <head>
                <title>Image and Prediction Display</title>
            </head>
            <body>
                <h1>Original Image and Prediction</h1>
                <div style="display: flex; justify-content: space-around;">
                    <div><img src="/static/{image_name}.png" alt="Original Image"></div>
                    <div><img src="/static/prediction_plot.png" alt="Prediction"></div>
                </div>
            </body>
        </html>
        """


        return HTMLResponse(html_content)
    except Exception as e:
        return {"error": f"An error occurred: {e}"}
    
