from analysis.dataloaders import CSVAnnotationDataloader, ImageDataloader
from analysis.fractal_funcs import DataAnalyzer
from analysis.geotif_analyzer import GeoTiffAnalyzer
from settings import CSV_FOLDER_PATH, MASKS_FOLDER_PATH, SOURCE_IMAGES_FOLDER_PATH, OUTPUT_FOLDER_PATH

def analyze_fractal():
    image_dataloader = ImageDataloader(MASKS_FOLDER_PATH)
    csv_dataloader = CSVAnnotationDataloader(CSV_FOLDER_PATH)

    analyzer = DataAnalyzer(image_dataloader, csv_dataloader)
    analyzer.analyze()

def analyze_geotiff():
    tif_analyzer = GeoTiffAnalyzer(SOURCE_IMAGES_FOLDER_PATH, OUTPUT_FOLDER_PATH)
    tif_analyzer.analyze()

def main():
    analyze_geotiff()
    analyze_fractal()


if __name__ == "__main__":
    main()