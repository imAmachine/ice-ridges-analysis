from analysis.dataloaders import CSVAnnotationDataloader, ImageDataloader
from analysis.fractal_funcs import DataAnalyzer, FractalAnalyzer
from analysis.funcs_koch import SnowKoch, SierpinskiTriangle, FractalTester
from settings import PHOTOS_FOLDER_PATH, CSV_FOLDER_PATH

def analize_data():
    image_dataloader = ImageDataloader(PHOTOS_FOLDER_PATH)
    csv_dataloader = CSVAnnotationDataloader(CSV_FOLDER_PATH)

    analyzer = DataAnalyzer(image_dataloader, csv_dataloader)
    analyzer.analyze()

def main():
    tester = FractalTester(SnowKoch, order=3, size=512, show_image=False, length=1.0)
    fractal_dimension = tester.test_fractal()
    print(f"Fractal Dimension (SnowKoch): {fractal_dimension:.5f}")

if __name__ == "__main__":
    main()