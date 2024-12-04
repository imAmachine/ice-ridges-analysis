from analysis.dataloaders import CSVAnnotationDataloader, ImageDataloader
from analysis.fractal_funcs import DataAnalyzer
from settings import PHOTOS_FOLDER_PATH, CSV_FOLDER_PATH


def main():
    image_dataloader = ImageDataloader(PHOTOS_FOLDER_PATH)
    csv_dataloader = CSVAnnotationDataloader(CSV_FOLDER_PATH)

    analyzer = DataAnalyzer(image_dataloader, csv_dataloader)
    analyzer.analyze(show_separately=False)


if __name__ == "__main__":
    main() # test