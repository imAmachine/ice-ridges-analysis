import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


class FractalAnalyzer:
    """
    Класс для анализа фрактальной размерности.
    """
    @staticmethod
    def box_counting(binary_image):
        """
        Реализация метода box counting.
        """
        sizes = []
        counts = []

        min_side = min(binary_image.shape)
        for size in range(2, min_side // 2 + 1, 2):
            count = 0
            for x in range(0, binary_image.shape[0] + 1, size):
                for y in range(0, binary_image.shape[1] + 1, size):
                    if np.any(binary_image[x:x+size, y:y+size] > 0):
                        count += 1
            
            sizes.append(size)
            counts.append(count)

        return sizes, counts

    @staticmethod
    def higuchi(data, k_max):
        if len(data) < k_max:
            raise ValueError("Length of spatial data is less than k_max.")

        L = []

        for k in range(1, k_max + 1):
            Lk = []

            for m in range(k):
                sum_distance = 0
                count = 0

                for i in range(1, (len(data) - m) // k + 1):
                    point1 = np.array(data[m + (i - 1) * k])
                    point2 = np.array(data[m + i * k])
                    sum_distance += np.linalg.norm(point2 - point1)
                    count += 1

                if count > 0:
                    normalization = (len(data) - 1) / (k * count * k)
                    Lk.append(sum_distance * normalization)

            if Lk:
                L.append(np.mean(Lk))

        return np.arange(1, k_max + 1), L

    @staticmethod
    def variance(data, scales):
        variances = []

        for scale in scales:
            if scale <= len(data):
                segments = len(data) // scale
                
                # Reshape the data into segments
                reshaped = np.array(data[:segments * scale]).reshape(segments, scale, 2)
                
                # Calculate local variances for each segment
                local_variances = []
                for segment in reshaped:
                    # Calculate pairwise distances within the segment
                    pairwise_distances = [
                        np.linalg.norm(segment[j] - segment[j+1]) 
                        for j in range(len(segment) - 1)
                    ]
                    local_variances.append(np.var(pairwise_distances))
                
                variances.append(np.mean(local_variances))

        return 1 / np.array(scales[:len(variances)]), variances

    @staticmethod
    def calculate_fractal_dimension(sizes, counts, epsilon=1e-10):
        log_sizes = np.log(np.array(sizes))
        log_counts = np.log(np.array(counts) + epsilon)
        slope, intercept, r_value, p_value, std_err = linregress(log_sizes, log_counts)
        return np.abs(slope)


class DataAnalyzer:
    """
    Класс для объединения всех компонентов и выполнения анализа.
    """

    def __init__(self, image_dataloader, csv_dataloader):
        self.image_dataloader = image_dataloader
        self.csv_dataloader = csv_dataloader

    def _analyze_images(self):
        """
        Анализ изображений для расчёта фрактальной размерности.
        """
        results = []

        logging.info("Начат анализ изображений.")
        for filename, binary_image in self.image_dataloader.get_all_data():
            sizes, counts = FractalAnalyzer.box_counting(binary_image)
            fractal_dimension = FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
            results.append({
                "filename": filename,
                "fractal_dimension": fractal_dimension
            })

        logging.info("Анализ изображений завершён.")
        return pd.DataFrame(results)

    def extract_coordinates(self, ridge_data):
        """
        Extract coordinate points from the ridge data
        
        Args:
        - ridge_data: DataFrame containing ridge information
        
        Returns:
        - List of coordinate tuples
        """
        # Assuming the coordinate column is stored as a string like "(x, y)"
        coordinates = ridge_data['Value of the first coordinate (X, Y)'].apply(
            lambda x: tuple(map(float, x.strip('()').split(',')))
        ).tolist()
        
        return coordinates
    
    def _analyze_time_series(self, k_max=10, scales=None):
        if scales is None:
            scales = [2, 4, 8, 16, 32]

        results = []

        for filename, data in self.csv_dataloader.get_all_data():
            try:
                # Extract coordinates
                coordinates = self.extract_coordinates(data)
                
                # Higuchi analysis
                if len(coordinates) >= k_max:
                    higguchi_a, higguchi_b = FractalAnalyzer.higuchi_spatial(coordinates, k_max=k_max)
                    higguchi_fractal_dimension = FractalAnalyzer.calculate_fractal_dimension(higguchi_a, higguchi_b)
                else:
                    higguchi_fractal_dimension = None
                
                # Variance analysis
                if len(coordinates) >= min(scales):
                    variance_a, variance_b = FractalAnalyzer.variance_spatial(coordinates, scales=scales)
                    variance_fractal_dimension = FractalAnalyzer.calculate_fractal_dimension(variance_a, variance_b)
                else:
                    variance_fractal_dimension = None
                
                # Store results
                results.append({
                    "filename": filename,
                    "higuchi_dimension": higguchi_fractal_dimension,
                    "variance_dimension": variance_fractal_dimension
                })

            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")

        return pd.DataFrame(results)

    def analyze(self):
        """
        Основной метод анализа, объединяющий обработку изображений и временных рядов.
        """
        # Анализ изображений
        images_results = self._analyze_images()
        print("\nРезультаты анализа изображений:")
        print(images_results.to_string(index=False, justify='center'))

        # Анализ временных рядов
        # time_series_results = self._analyze_time_series()
        # print("\nРезультаты анализа пространственных рядов:")
        # print(time_series_results.to_string(index=False, justify='center'))

        return images_results#, time_series_results
