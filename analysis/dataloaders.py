import os
import pandas as pd
import cv2


class ImageDataloader:
    """
    Класс для работы с набором изображений.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = self._get_image_files()

    def _get_image_files(self):
        """
        Возвращает список изображений в папке.
        """
        return [f for f in os.listdir(self.folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def load_image(self, filename):
        """
        Загружает изображение по имени файла и преобразует его в бинарный формат.
        """
        path = os.path.join(self.folder_path, filename)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {filename}")
        _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return binary_image

    def get_all_data(self):
        """
        Генерирует бинарные изображения из всех файлов в папке.
        """
        for filename in self.image_files:
            yield filename, self.load_image(filename)


class CSVAnnotationDataloader:
    """
    Класс для работы с разметкой в CSV-файлах.
    """
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.csv_files = self._get_csv_files()

    def _get_csv_files(self):
        """
        Возвращает список CSV-файлов в папке.
        """
        return [f for f in os.listdir(self.folder_path) if f.endswith('.csv')]

    def load_csv(self, filename):
        """
        Загружает содержимое CSV-файла в DataFrame.
        """
        path = os.path.join(self.folder_path, filename)
        return pd.read_csv(path)

    def get_all_data(self):
        """
        Генерирует DataFrame для всех CSV-файлов в папке.
        """
        for filename in self.csv_files:
            yield filename, self.load_csv(filename)