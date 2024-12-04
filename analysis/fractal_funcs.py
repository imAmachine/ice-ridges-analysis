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

    def _visualize(self, log_sizes, log_counts, filename, show_separately=True, ax=None):
        """
        Визуализация с возможностью выбора вывода графиков по отдельности или все вместе.
        Подпись на графике будет только с названием файла.
        """
        if show_separately:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(log_sizes, log_counts, 'o-')
            ax.set_xlabel("log(1/r)")
            ax.set_ylabel("log(N(r))")
            ax.set_title(filename)
            plt.show()
        else:
            ax.plot(log_sizes, log_counts, 'o-')
            ax.set_xlabel("log(1/r)")
            ax.set_ylabel("log(N(r))")
            ax.set_title(filename)

        return ax

    def _create_combined_plot(self, log_data, method_name):
        """
        Метод для объединённого отображения графиков в сетке.
        Название метода выводится один раз в общем заголовке.
        """
        n_plots = len(log_data)
        n_cols = 5  # Задаём количество столбцов
        n_rows = (n_plots + 1) // n_cols  # Расчитываем количество строк

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
        fig.suptitle(method_name, fontsize=16)  # Общий заголовок для всех графиков

        # Если у нас только один график, ax будет не список, а один объект
        if n_rows == 1:
            axes = [axes]

        # Проходим по всем данным и отображаем графики
        for idx, (log_sizes, log_counts, filename) in enumerate(log_data):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[col]
            ax.plot(log_sizes, log_counts, 'o-', markersize=2)
            ax.set_xlabel("log(1/r)")
            ax.set_ylabel("log(N(r))")
            ax.set_title(filename)

        # Убираем пустые подграфики, если количество графиков не делится нацело на количество столбцов
        for idx in range(n_plots, n_rows * n_cols):
            fig.delaxes(axes.flatten()[idx])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Оставляем место для заголовка
        plt.show()

    def _analyze_images(self, show_separately=True):
        """
        Анализ изображений и визуализация фрактальной размерности.
        """
        results = []
        log_data = []

        print("Анализ изображений:")
        for idx, (filename, binary_image) in enumerate(self.image_dataloader.get_all_images()):
            sizes, counts = FractalAnalyzer.box_counting(binary_image)
            fractal_dimension, log_sizes, log_counts = FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
            results.append((filename, fractal_dimension))

            print(f"Фрактальная размерность для {filename}: {fractal_dimension}")
            if show_separately:
                self._visualize(log_sizes, log_counts, filename=f"Фрактальная размерность ({filename})", show_separately=True)
            else:
                log_data.append((log_sizes, log_counts, filename))

        if not show_separately:
            self._create_combined_plot(log_data, 'Фрактальная размерность')

        return results

    def _analyze_time_series(self, show_separately=True):
        """
        Анализ временных рядов и визуализация методов Хигучи и дисперсии.
        """
        results = []
        log_data_higuchi = []
        log_data_variance = []

        print("Анализ временных рядов:")
        for filename, data in self.csv_dataloader.get_all_csv():
            time_series = data.iloc[:, 0].values
            higuchi_dimension, log_k, log_L = FractalAnalyzer.higuchi(time_series, k_max=10)
            variance_dimension, log_scales, log_variances = FractalAnalyzer.variance(time_series, scales=[2, 4, 8, 16, 32])

            results.append((filename, higuchi_dimension, variance_dimension))

            print(f"Метод Хигучи для {filename}: {higuchi_dimension}")
            print(f"Метод дисперсии для {filename}: {variance_dimension}")

            if show_separately:
                self._visualize(log_k, log_L, filename=f"Метод Хигучи ({filename})", show_separately=True)
                self._visualize(log_scales, log_variances, filename=f"Метод дисперсии ({filename})", show_separately=True)
            else:
                log_data_higuchi.append((log_k, log_L, filename))
                log_data_variance.append((log_scales, log_variances, filename))

        if not show_separately:
            self._create_combined_plot(log_data_higuchi, 'Метод Хигучи')
            self._create_combined_plot(log_data_variance, 'Метод дисперсии')

        return results

    def analyze(self, show_separately=True):
        """
        Основной метод анализа.
        """
        images_results = self._analyze_images(show_separately)
        time_series_results = self._analyze_time_series(show_separately)

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