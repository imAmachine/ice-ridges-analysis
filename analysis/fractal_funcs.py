import logging
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


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

        for size in range(1, binary_image.shape[0] // 2, 2):
            resized = cv2.resize(binary_image, (size, size), interpolation=cv2.INTER_NEAREST)
            count = np.sum(resized > 0)
            sizes.append(size)
            counts.append(count)

        return sizes, counts

    @staticmethod
    def higuchi(data, k_max):
        """
        Реализация метода Хигучи для временных рядов.
        """
        L = []
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(k):
                sum_diff = 0
                for i in range(1, (len(data) - m) // k):
                    sum_diff += abs(data[m + i * k] - data[m + (i - 1) * k])
                normalization = (len(data) - 1) / (k * (len(data) - m))
                Lk.append(sum_diff * normalization)
            L.append(np.mean(Lk))

        log_k = np.log(np.arange(1, k_max + 1))
        log_L = np.log(L)
        coefficients = np.polyfit(log_k, log_L, 1)
        fractal_dimension = -coefficients[0]
        
        return fractal_dimension, log_k, log_L

    @staticmethod
    def variance(data, scales):
        """
        Реализация метода дисперсии (variance) для фрактальной размерности.
        """
        variances = []
        for scale in scales:
            segments = len(data) // scale
            reshaped = data[:segments * scale].reshape(segments, scale)
            local_variances = np.var(reshaped, axis=1)
            variances.append(np.mean(local_variances))

        log_scales = np.log(1 / np.array(scales))
        log_variances = np.log(variances)
        coefficients = np.polyfit(log_scales, log_variances, 1)
        fractal_dimension = -coefficients[0] / 2
        
        return fractal_dimension, log_scales, log_variances

    @staticmethod
    def calculate_fractal_dimension(sizes, counts):
        """
        Вычисляет фрактальную размерность по данным.
        """
        sizes, counts = zip(*[(s, c) for s, c in zip(sizes, counts) if c > 0])

        log_sizes = np.log(1 / np.array(sizes))
        log_counts = np.log(np.array(counts))
        coefficients = np.polyfit(log_sizes, log_counts, 1)
        fractal_dimension = -coefficients[0]
        return fractal_dimension, log_sizes, log_counts


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
        for filename, binary_image in self.image_dataloader.get_all_images():
            sizes, counts = FractalAnalyzer.box_counting(binary_image)
            fractal_dimension, _, _ = FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
            results.append({
                "filename": filename,
                "fractal_dimension": fractal_dimension
            })

        logging.info("Анализ изображений завершён.")
        return pd.DataFrame(results)

    def _analyze_time_series(self, k_max=10, scales=None):
        """
        Анализ временных рядов методами Хигучи и дисперсии.
        """
        if scales is None:
            scales = [2, 4, 8, 16, 32]

        results = []

        logging.info("Начат анализ временных рядов.")
        for filename, data in self.csv_dataloader.get_all_csv():
            try:
                # Параметры для анализа
                params = {
                    "Area": data["Area"],
                    "Length": data["Length"],
                    "Width": data["Width"],
                    "Ridge orientation angle": data["Ridge orientation angle"]
                }

                # Анализ для каждого параметра
                for param_name, time_series in params.items():
                    # Метод Хигучи
                    higuchi_dimension, _, _ = FractalAnalyzer.higuchi(time_series.values, k_max=k_max)

                    # Метод дисперсии
                    variance_dimension, _, _ = FractalAnalyzer.variance(time_series.values, scales=scales)

                    # Сохранение результатов
                    results.append({
                        "filename": filename,
                        "parameter": param_name,
                        "higuchi_dimension": higuchi_dimension,
                        "variance_dimension": variance_dimension
                    })

            except Exception as e:
                logging.error(f"Ошибка при обработке файла {filename}: {e}")

        logging.info("Анализ временных рядов завершён.")
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
        time_series_results = self._analyze_time_series()
        print("\nРезультаты анализа временных рядов:")
        print(time_series_results.to_string(index=False, justify='center'))

        return images_results, time_series_results