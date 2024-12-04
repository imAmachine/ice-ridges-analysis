import numpy as np
import cv2
import matplotlib.pyplot as plt


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

    def _visualize(filename, log_sizes, log_counts):
        plt.plot(log_sizes, log_counts, 'o-')
        plt.xlabel("log(1/r)")
        plt.ylabel("log(N(r))")
        plt.title(f"Фрактальная размерность: {filename}")
        plt.show()

    def _images_analyze(self):
        """
        Анализ изображений на фрактальную размерность.
        """
        results = []

        print("Анализ изображений:")
        for filename, binary_image in self.image_dataloader.get_all_images():
            sizes, counts = FractalAnalyzer.box_counting(binary_image)
            fractal_dimension, log_sizes, log_counts = FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
            results.append((filename, fractal_dimension))

            print(f"Фрактальная размерность для {filename}: {fractal_dimension}")
            self._visualize(filename, log_sizes, log_counts, "Фрактальная размерность (изображение)")

        return results

    def _time_series_analyze(self):
        """
        Анализ временных рядов из CSV-файлов.
        """
        results = []

        print("Анализ временных рядов:")
        for filename, data in self.csv_dataloader.get_all_csv():
            time_series = data.iloc[:, 0].values
            higuchi_dimension, log_k, log_L = FractalAnalyzer.higuchi(time_series, k_max=10)
            variance_dimension, log_scales, log_variances = FractalAnalyzer.variance(time_series, scales=[2, 4, 8, 16, 32])
            
            results.append((filename, higuchi_dimension, variance_dimension))

            print(f"Метод Хигучи для {filename}: {higuchi_dimension}")
            print(f"Метод дисперсии для {filename}: {variance_dimension}")

            self._visualize(filename, log_k, log_L, "Метод Хигучи")
            self._visualize(filename, log_scales, log_variances, "Метод дисперсии")

        return results

    def analyze(self):
        """
        Основной метод анализа.
        """
        images_results = self._images_analyze()
        time_series_results = self._time_series_analyze()

        # Вывод результатов анализа изображений
        print("\nРезультаты анализа изображений:")
        print(f"{'Файл':<30} | {'Фрактальная размерность':>20}")
        print("-" * 55)
        for file, dimension in images_results:
            print(f"{file:<30} | {dimension:>20.4f}")

        # Вывод результатов анализа временных рядов
        print("\nРезультаты анализа временных рядов:")
        print(f"{'Файл':<30} | {'Метод Хигучи':>15} | {'Метод дисперсии':>15}")
        print("-" * 65)
        for file, higuchi_dim, variance_dim in time_series_results:
            print(f"{file:<30} | {higuchi_dim:>15.4f} | {variance_dim:>15.4f}")