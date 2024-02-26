"""
This module creates image masks based on polygon annotations
provided in a JSON file and store them into specific folders.
"""

import json
import os
import  matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from utils import DirectoryManager



class ImageMaskGenerator:
    """
    This class generates image masks based on polygon annotations
    provided in a JSON file.
    """

    def __init__(self, json_path, images_folder, masks_folder, YN = False):
        """
        Initializes the image mask generator.

        :param json_path: Path to the JSON file containing the annotations.
        :param images_folder: Folder containing the original images.
        :param masks_folder: Destination folder for saving the generated masks.
        :param YN: Boolean indicating whether we use another json format coming from Label Studio.
        """
        self.json_path = json_path
        self.images_folder = images_folder
        self.masks_folder = masks_folder
        self.YN = YN

    def load_json_data(self):
        """
        Loads JSON data from the specified file.

        :return: The loaded JSON data.
        """
        with open(self.json_path) as file:
            return json.load(file)

    def generate_masks(self):
        """
        Generates masks for each image based on the polygon annotations
        in the JSON file.
        """
        data = self.load_json_data()

        for label_info in data:
            if self.YN :
                image_name = label_info.get('data').get('image').split('-')[1]
            else :
                image_name = label_info['data_row']['external_id']
            image_path = os.path.join(self.images_folder, image_name)
            image = Image.open(image_path)

            mask = self.create_mask(image, label_info)
            plt.imshow(mask)
            self.save_mask(mask, image_name)

    def create_mask(self, image, label_info):
        """
        Creates a mask for a single image based on the polygon annotations.

        :param image: The PIL Image object of the image.
        :param label_info: The labeling information for this image.
        :return: The PIL Image object of the mask.
        """
        width, height = image.size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        if self.YN :
            polygons = []
            for i in range(len(label_info.get('annotations')[0].get('result'))):
                polygons.append(label_info.get('annotations')[0].get('result')[i].get('value').get('points'))

            for polygon in polygons:
                points = [(point[0]*width/100,
                point[1]*height/100) for point in polygon]
                draw.polygon(points, fill=255)

        else :
            polygons = [obj["polygon"] for obj in
            label_info["projects"]["clqxivkqq03v807wi5dp3e5yk"]["labels"][0]["annotations"]["objects"]]
            for polygon in polygons:
                points = [(point['x'], point['y']) for point in polygon]
                draw.polygon(points, fill=255)

        return mask

    def save_mask(self, mask, image_name):
        """
        Saves the mask in the specified folder.

        :param mask: The PIL Image object of the mask to save.
        :param image_name: The name of the source image.
        """
        mask_file_name = os.path.basename(image_name).replace('.png', '_mask.png')
        mask_path = os.path.join(self.masks_folder, mask_file_name)
        # create directory if not exists
        directory_manager = DirectoryManager()
        directory_manager.ensure_directory_exists(mask_path)
        mask.save(mask_path)
        print(f"Masque sauvegardé dans : {mask_path}")

if __name__ == '__main__' :

    yn_generator = ImageMaskGenerator('../json/mask.json',
                                '../data/images',
                                '../data/masks',
                                YN = True)
    yn_generator.generate_masks()

    mr_generator = ImageMaskGenerator('../json/mask_maxime.ndjson',
                                '../data/images',
                                '../data/masks',
                                YN = False)
    mr_generator.generate_masks()

    yn_generator_test_data = ImageMaskGenerator('../json/mask_test.json',
                                '../data/test/images',
                                '../data/test/masks',
                                YN = True)
    yn_generator_test_data.generate_masks()
