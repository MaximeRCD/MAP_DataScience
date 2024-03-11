"""
This script sets up a FastAPI application to detect free parking spaces from images provided via URL. 
It employs a pre-trained model for prediction, uses image processing for preparing and analyzing images, 
and serves the results through a simple web interface.

Modules and packages used include FastAPI for the web framework, OpenCV for image processing, 
Albumentations for image transformations, PyTorch utilities for model processing, and several standard 
libraries for file and system operations.

The application has two main endpoints: a GET request to display a simple form for inputting the image URL 
and a POST request to process the submitted image and display the prediction results.
"""

import os
import glob
import cv2
import sys
import matplotlib
import matplotlib.pyplot as plt
import requests
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from constants import PARAMS, PRETRAINED_MODEL_PATH, API_IMAGES_DIR
from datasets import FreeParkingPlacesInferenceDataset
from test_model import load_model, predict
from utils import Visualizer, DirectoryManager

matplotlib.use('Agg')


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
    Generates predictions for free parking spaces in an image using a pre-trained model.

    This function preprocesses the image located at the given filename path according to the model's
    requirements, performs the prediction, and processes the prediction result to generate masks indicating
    free parking spaces.

    Args:
        filename (str): Path to the image file on which to perform the prediction.

    Returns:
        List[np.ndarray]: A list of prediction masks corresponding to the input image, resized to match the
        original image dimensions.
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

# Creation of the FastAPI application
app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def serve():
    """
    Serves the main page of the application, presenting a form where users can input the URL of an image to
    be processed for free parking space detection.

    This endpoint responds to GET requests with a simple HTML form designed for URL submission.

    Returns:
        HTMLResponse: An HTML document containing the input form.
    """

    return """
    <html>
        <head>
            <title>Image Display</title>
        </head>
        <body>
            <h1>Parking places destection model</h1>
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
    This endpoint fetches the image from the provided URL, save it as a PNG image in a specific folder,
    calls the model to predict and save the prediction in order to display the results.The final
    visualization is served in an HTML page.

    Args:
        image_url (str): The URL of the image submitted through the form.

    Returns:
        HTMLResponse: An HTML document displaying the original image alongside its prediction visualization.
    """

    #folder where you want to save the image
    save_folder = Path(API_IMAGES_DIR)
    print(save_folder)
    #DirectoryManager.ensure_directory_exists(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist
    app.mount("/static", StaticFiles(directory=API_IMAGES_DIR), name="static")
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
        saved_image = cv2.imread(str(save_path))
        saved_image = cv2.cvtColor(saved_image, cv2.COLOR_BGR2RGB)
        save_name = "prediction_plot.png"
        plot_path = save_folder / save_name
    
        Visualizer.visualize(
            image=saved_image, 
            mask=pred[0], 
            original_image=None, 
            original_mask=None, 
            title="Prediction Visualization"
        )
    
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        html_content = f"""
        <html>
            <head>
                <title>Image and Prediction Display</title>
            </head>
            <body>
                <h1>Original Image and Prediction</h1>
                <div style="display: flex; justify-content: space-around;">
                    <div><img src="/static/{save_name}" alt="Prediction Visualization"></div>
                </div>
            </body>
        </html>
        """


        return HTMLResponse(html_content)
    except Exception as e:
        return {"error": f"An error occurred: {e}"}
    
