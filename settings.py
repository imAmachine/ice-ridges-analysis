from dotenv import load_dotenv
import os

load_dotenv()

CSV_FOLDER_PATH = os.getenv('CSV_FOLDER_PATH')
PHOTOS_FOLDER_PATH = os.getenv('PHOTOS_FOLDER_PATH')