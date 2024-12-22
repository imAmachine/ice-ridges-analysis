import os
from analysis.dataloaders import CSVAnnotationDataloader, ImageDataloader
from analysis.fractal_funcs import DataAnalyzer, FractalAnalyzer
from analysis.funcs_koch import SnowKoch, SierpinskiTriangle, FractalTester
from analysis.geotif_analyzer import GeoTiffAnalyzer
from analysis.interpolation_processor import InterpolationProcessor
from settings import CSV_FOLDER_PATH, MASKS_FOLDER_PATH, SOURCE_IMAGES_FOLDER_PATH, ANALYSIS_OUTPUT_FOLDER_PATH

geo_data_path = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'geo_data.csv')
distances_csv_path = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'distances.csv')
unified_images_folder_path = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'unified_images/')

def analyze_fractal():
    image_dataloader = ImageDataloader(MASKS_FOLDER_PATH)
    csv_dataloader = CSVAnnotationDataloader(CSV_FOLDER_PATH)
    analyzer = DataAnalyzer(image_dataloader, csv_dataloader)
    analyzer.analyze()

def analyze_geotiff():
    tif_analyzer = GeoTiffAnalyzer(SOURCE_IMAGES_FOLDER_PATH, ANALYSIS_OUTPUT_FOLDER_PATH)
    tif_analyzer.analyze()

def interpolation_tiff():
    processor = InterpolationProcessor(SOURCE_IMAGES_FOLDER_PATH, unified_images_folder_path, distances_csv_path)
    processor.process(geo_data_path=geo_data_path, groups={
        'group_1': ('ridge_8.tif', 'ridge_9.tif', 'ridge_10.tif'),
        'group_2': ('ridge_2.tif', 'ridge_3.tif'),
        'group_3': ('ridge_4.tif', 'ridge_5.tif', 'ridge_6.tif', 'ridge_7.tif'),
    })

def main():
    analyze_geotiff()
    interpolation_tiff()
    # analyze_fractal()
    # tester = FractalTester(SnowKoch, order=3, size=512, show_image=False, length=1.0)
    # fractal_dimension = tester.test_fractal()
    # print(f"Fractal Dimension (SnowKoch): {fractal_dimension:.5f}")

if __name__ == "__main__":
    main()