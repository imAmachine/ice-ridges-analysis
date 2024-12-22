from analysis.dataloaders import CSVAnnotationDataloader, ImageDataloader
from analysis.fractal_funcs import DataAnalyzer, FractalAnalyzer
from analysis.funcs_koch import SnowKoch, SierpinskiTriangle, FractalTester
from analysis.geotif_analyzer import GeoTiffAnalyzer, InterpolationProcessor
from settings import CSV_FOLDER_PATH, MASKS_FOLDER_PATH, SOURCE_IMAGES_FOLDER_PATH, OUTPUT_FOLDER_PATH, UNIFIED_IMAGES_FOLDER_PATH, DISTANCES_CSV_PATH

def analyze_fractal():
    image_dataloader = ImageDataloader(MASKS_FOLDER_PATH)
    csv_dataloader = CSVAnnotationDataloader(CSV_FOLDER_PATH)
    analyzer = DataAnalyzer(image_dataloader, csv_dataloader)
    analyzer.analyze()

def analyze_geotiff():
    tif_analyzer = GeoTiffAnalyzer(SOURCE_IMAGES_FOLDER_PATH, OUTPUT_FOLDER_PATH)
    tif_analyzer.analyze()

def interpolation_tiff():
    processor = InterpolationProcessor(SOURCE_IMAGES_FOLDER_PATH, UNIFIED_IMAGES_FOLDER_PATH, DISTANCES_CSV_PATH)
    processor.process()

def main():
    # analyze_geotiff()
    interpolation_tiff()
    # analyze_fractal()
    # tester = FractalTester(SnowKoch, order=3, size=512, show_image=False, length=1.0)
    # fractal_dimension = tester.test_fractal()
    # print(f"Fractal Dimension (SnowKoch): {fractal_dimension:.5f}")

if __name__ == "__main__":
    main()