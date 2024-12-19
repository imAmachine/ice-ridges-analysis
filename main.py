from analysis.dataloaders import CSVAnnotationDataloader, ImageDataloader
from analysis.fractal_funcs import DataAnalyzer, FractalAnalyzer
from analysis.funcs_koch import SnowKoch, SierpinskiTriangle
from settings import PHOTOS_FOLDER_PATH, CSV_FOLDER_PATH
import matplotlib.pyplot as plt
import numpy as np

def analize_data():
    image_dataloader = ImageDataloader(PHOTOS_FOLDER_PATH)
    csv_dataloader = CSVAnnotationDataloader(CSV_FOLDER_PATH)

    analyzer = DataAnalyzer(image_dataloader, csv_dataloader)
    analyzer.analyze()

def test_koch_snowflake(order=3, size=512, length=1.0):
    """Тестирование метода box counting."""
    koch = SnowKoch(order, size, length)
    binary_image = koch.get_binary()
    sizes, counts = FractalAnalyzer.box_counting(binary_image)
    fractal_dimension = FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
    SnowKoch._visualize_snowflake(order, size, length)
    plot_log_counts_vs_sizes(sizes, counts)
    return fractal_dimension

def test_sierpinski(order=5, size=512):
    triangle = SierpinskiTriangle(order, size)
    binary_image = triangle.get_binary()
    sizes, counts = FractalAnalyzer.box_counting(binary_image)
    fractal_dimension = FractalAnalyzer.calculate_fractal_dimension(sizes, counts)
    SierpinskiTriangle._visualize_serpinski(order, size)
    plot_log_counts_vs_sizes(sizes, counts)
    return fractal_dimension

def plot_log_counts_vs_sizes(sizes, counts):
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

def resolution_test(order=3, length=1.0, sizes=[128, 256, 512, 1024, 2048]):
    """Тест влияния разрешения и длины на размерность."""
    for size in sizes:
        fractal_dimension = test_koch_snowflake(order, size, length)
        print(f"Size: {size}, Length: {length}, Fractal Dimension: {fractal_dimension:.5f}")

def main():
    print(f"Фрактальная размерность снежинки Коха: {test_koch_snowflake():.5f}")
    print(f"Фрактальная размерность треугольника Серпинского: {test_sierpinski():.5f}")
    # resolution_test(order=3)  # Тест разрешения

if __name__ == "__main__":
    main()