import os
from analysis.dataloaders import CSVAnnotationDataloader, ImageDataloader
from analysis.fractal_funcs import DataAnalyzer, FractalAnalyzer
from analysis.funcs_koch import SnowKoch, SierpinskiTriangle, FractalTester
from analysis.geotif_analyzer import GeoTiffAnalyzer
from analysis.interpolation_processor import InterpolationProcessor
from settings import CSV_FOLDER_PATH, MASKS_FOLDER_PATH, SOURCE_IMAGES_FOLDER_PATH, ANALYSIS_OUTPUT_FOLDER_PATH

unified_images_path = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'unified_analyze/images')

def analyze_fractal():
    image_dataloader = ImageDataloader(MASKS_FOLDER_PATH)
    csv_dataloader = CSVAnnotationDataloader(CSV_FOLDER_PATH)
    analyzer = DataAnalyzer(image_dataloader, csv_dataloader)
    analyzer.analyze()

def analyze_geotiff(analyze_name, images_folder_path, analysis_output_folder_path,
                    distances=False, visualize_pixel_size_barplot=True):
    tif_analyzer = GeoTiffAnalyzer(images_folder_path, analysis_output_folder_path)
    tif_analyzer.analyze(analyze_name, distances, visualize_pixel_size_barplot)

def interpolation_tiff():
    source_geo_data_path = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'geo_data.csv')
    source_dist_csv_path = os.path.join(ANALYSIS_OUTPUT_FOLDER_PATH, 'distances.csv')
    
    processor = InterpolationProcessor(SOURCE_IMAGES_FOLDER_PATH, unified_images_path, source_dist_csv_path)
    processor.process(geo_data_path=source_geo_data_path, groups={
        'group_1': ('ridge_8.tif', 'ridge_9.tif', 'ridge_10.tif'),
        'group_2': ('ridge_2.tif', 'ridge_3.tif'),
        'group_3': ('ridge_4.tif', 'ridge_5.tif', 'ridge_6.tif', 'ridge_7.tif'),
    })

def main():
    analyze_geotiff('source_analyze', SOURCE_IMAGES_FOLDER_PATH, ANALYSIS_OUTPUT_FOLDER_PATH, 
                    distances=True, visualize_pixel_size_barplot=True) # первоначальный анализ исходных geotif
    # interpolation_tiff() # унификация с помощью интерполяции
    
    analyze_geotiff('unified_analyze', unified_images_path, ANALYSIS_OUTPUT_FOLDER_PATH,
                    distances=False, visualize_pixel_size_barplot=True) # анализ интерполированных geotif
    
    
    # analyze_fractal()
    # tester = FractalTester(SnowKoch, order=3, size=512, show_image=False, length=1.0)
    # fractal_dimension = tester.test_fractal()
    # print(f"Fractal Dimension (SnowKoch): {fractal_dimension:.5f}")

if __name__ == "__main__":
    main()