from analysis.dataloaders import CSVAnnotationDataloader, ImageDataloader
from analysis.fractal_funcs import DataAnalyzer
from settings import PHOTOS_FOLDER_PATH, CSV_FOLDER_PATH


def main():
    image_dataset = ImageDataloader(PHOTOS_FOLDER_PATH)
    csv_dataset = CSVAnnotationDataloader(CSV_FOLDER_PATH)
    analyzer = DataAnalyzer(image_dataset, csv_dataset)
    analyzer.analyze()


if __name__ == "__main__":
    main() # test