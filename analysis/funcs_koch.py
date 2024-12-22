import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.stats import linregress
from .fractal_funcs import FractalAnalyzer

class SnowKoch:
    def __init__(self, order=3, size=512, length=1.0):
        self.order = order
        self.size = size
        self.length = length
        self.image = None
        self.cropped_image = None

    def _generate_koch_snowflake(self):
        """Генерация точек снежинки Коха."""
        def koch_curve(points, order):
            if order == 0:
                return points
            new_points = []
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                a = p1
                b = p1 + (p2 - p1) / 3
                d = p1 + 2 * (p2 - p1) / 3
                e = p2
                c = b + (d - b) @ np.array([[0.5, -np.sqrt(3) / 2],
                                            [np.sqrt(3) / 2, 0.5]])
                new_points += [a, b, c, d]
            new_points.append(points[-1])
            return koch_curve(np.array(new_points), order - 1)

        p0 = np.array([0, 0])
        p1 = np.array([self.length, 0])
        p2 = np.array([self.length / 2, self.length * np.sqrt(3) / 2])
        initial_points = np.array([p0, p1, p2, p0])
        return koch_curve(initial_points, self.order)

    def _crop_to_content(self, image):
        """Обрезка изображения до непустого содержимого."""
        coords = cv2.findNonZero(image)
        x, y, w, h = cv2.boundingRect(coords)
        cropped_image = image[y:y + h, x:x + w]
        return cropped_image

    def get_binary(self):
        """Создаёт бинарное изображение снежинки."""
        snowflake = self._generate_koch_snowflake()
        image = np.zeros((self.size, self.size), dtype=np.uint8)

        # Нормализация координат
        normalized_snowflake = (snowflake - snowflake.min()) / (snowflake.max() - snowflake.min())
        pixel_coords = (normalized_snowflake * (self.size - 1)).astype(int)

        # Рисование снежинки
        for i in range(len(pixel_coords) - 1):
            cv2.line(image, tuple(pixel_coords[i]), tuple(pixel_coords[i + 1]), color=255, thickness=1)

        self.image = image
        self.cropped_image = self._crop_to_content(image)
        return self.cropped_image

    def _visualize_snowflake(order=3, size=512, length=1.0):
        """Визуализация снежинки для проверки влияния length."""
        koch = SnowKoch(order, size, length)
        binary_image = koch.get_binary()
        plt.imshow(binary_image, cmap='gray')
        plt.title(f"Order: {order}, Size: {size}, Length: {length}")
        plt.show()

class SierpinskiTriangle:
    def __init__(self, order=5, size=512, length=1):
        self.order = order
        self.size = size
        self.image = None

    def _generate_triangle(self, points, order):
        if order == 0:
            return [points]
        else:
            midpoints = [(points[i] + points[(i + 1) % 3]) // 2 for i in range(3)]
            return (
                    self._generate_triangle([points[0], midpoints[0], midpoints[2]], order - 1) +
                    self._generate_triangle([midpoints[0], points[1], midpoints[1]], order - 1) +
                    self._generate_triangle([midpoints[2], midpoints[1], points[2]], order - 1)
            )

    def get_binary(self):
        """Создаёт бинарное изображение треугольника Серпинского."""
        size = self.size
        image = np.zeros((size, size), dtype=np.uint8)
        p0, p1, p2 = [0, size - 1], [size // 2, 0], [size - 1, size - 1]
        triangles = self._generate_triangle(np.array([p0, p1, p2]), self.order)
        for tri in triangles:
            pts = np.array(tri).reshape((-1, 1, 2)).astype(np.int32)  # Преобразование списка в массив
            cv2.fillPoly(image, [pts], 255)
        self.image = image
        return image

    def _visualize_serpinski(order=5, size=512):
        """Визуализация снежинки для проверки влияния length."""
        sierpinski = SierpinskiTriangle(order, size)
        binary_image = sierpinski.get_binary()
        plt.imshow(binary_image, cmap='gray')
        plt.title(f"Order: {order}, Size: {size}")
        plt.show()

class FractalTester:
    def __init__(self, fractal_class=SnowKoch, order=3, size=512, show_image=False, **kwargs):
        """
        Класс для проверки фрактальных свойств.

        :param fractal_class: Класс фрактала (например, SnowKoch или SierpinskiTriangle).
        :param order: Порядок фрактала.
        :param size: Размер изображения.
        :param show_image: Показывать ли изображение фрактала.
        :param kwargs: Дополнительные параметры для инициализации фрактала.
        """
        self.fractal = fractal_class(order=order, size=size, **kwargs)
        self.order = order
        self.size = size
        self.show_image = show_image

    def test_fractal(self):
        """Проверка фрактала с использованием метода box counting."""
        binary_image = self.fractal.get_binary()

        if self.show_image:
            self._show_fractal_image(binary_image)

        sizes, counts = FractalAnalyzer.box_counting(binary_image)
        fractal_dimension = FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
        self._plot_log_counts_vs_sizes(sizes, counts)
        quality_metrics = self._evaluate_approximation_quality(sizes, counts)
        print(f"R^2: {quality_metrics['r_squared']:.5f}")
        print(f"Standard Deviation: {quality_metrics['std_dev']:.5f}")
        return fractal_dimension

    def _show_fractal_image(self, binary_image):
        """Отображение изображения фрактала."""
        plt.imshow(binary_image, cmap='gray')
        plt.title(f"Fractal (Order: {self.order}, Size: {self.size})")
        plt.axis('off')
        plt.show()

    @staticmethod
    def _plot_log_counts_vs_sizes(sizes, counts):
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)

        plt.figure(figsize=(8, 6))
        plt.plot(log_sizes, log_counts, 'o-', label='Box Counting Data')
        plt.title("Log(Counts) vs. Log(Sizes)")
        plt.xlabel("Log(Sizes)")
        plt.ylabel("Log(Counts)")
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def _evaluate_approximation_quality(sizes, counts):
        """
        Расчет коэффициента детерминации (R^2) и стандартного отклонения.
        """
        log_sizes = np.log(np.array(sizes))
        log_counts = np.log(np.array(counts))

        slope, intercept, r_value, p_value, std_err = linregress(log_sizes, log_counts)

        # Предсказанные значения
        predicted_counts = slope * log_sizes + intercept

        # Остатки (разности между реальными и предсказанными значениями)
        residuals = log_counts - predicted_counts

        # Стандартное отклонение
        std_dev = np.std(residuals)

        return {
            'r_squared': r_value ** 2,
            'std_dev': std_dev,
        }