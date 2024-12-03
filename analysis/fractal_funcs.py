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
        Анализ изображений на фрактальную размерность
        """
        results = []

        print("Анализ изображений:")
        for filename, binary_image in self.image_dataloader.get_all_images():
            sizes, counts = FractalAnalyzer.box_counting(binary_image)
            fractal_dimension, log_sizes, log_counts = FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
            results.append((filename, fractal_dimension))

            print(f"Фрактальная размерность для {filename}: {fractal_dimension}")
            self._visualize(filename, log_sizes, log_counts)

        return results


    def analyze(self):
        """
        Основной метод анализа.
        """
        images_results = self._images_analyze()

        ### CODE WIP ###

        # Анализ CSV-файлов 
        # print("Анализ CSV-файлов:")
        # for filename, data in self.csv_dataloader.get_all_csv():
        #     print(f"Обработан файл разметки: {filename}")
        #     print(data.head())

        # Вывод результатов
        print("\nРезультаты анализа изображений:")
        print(f"{'Файл':<30} | {'Фрактальная размерность':>20}")
        print("-" * 55)
        for file, dimension in images_results:
            print(f"{file:<30} | {dimension:>20.4f}")