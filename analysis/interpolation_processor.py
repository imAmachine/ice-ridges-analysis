import os
import pandas as pd
from osgeo import gdal


class InterpolationProcessor:
    def __init__(self, input_dir, output_dir, csv_file, interpolation_method='bilinear'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.csv_file = csv_file
        self.interpolation_method = interpolation_method

    def calculate_average_pixel_size(self, group_data: pd.DataFrame):
        """
        Рассчитывает средний размер пикселя (xRes, yRes) для группы изображений.
        """
        avg_pixel_deg_width = group_data['pixel_size_x_deg'].mean()
        avg_pixel_deg_height = group_data['pixel_size_y_deg'].mean()
        return avg_pixel_deg_width, avg_pixel_deg_height

    def interpolate_file(self, avg_pixel_deg_size, file, output_file):
        try:
            print(f'Происходит интерполяция {file}')
            gdal.Warp(
                output_file,
                file,
                format="GTiff",
                xRes=avg_pixel_deg_size[0],
                yRes=avg_pixel_deg_size[1],
                resampleAlg=self.interpolation_method,
                creationOptions=["COMPRESS=LZW"]
            )
            output_dataset = gdal.Open(output_file)
            if output_dataset:
                print(f"Размеры файла {output_file}: {output_dataset.RasterXSize}x{output_dataset.RasterYSize}")
                print(f"Размер пикселя: {avg_pixel_deg_size[0]}x{avg_pixel_deg_size[1]}")
                output_dataset = None
            else:
                print(f"Не удалось открыть обработанный файл: {output_file}")
            print(f"Интерполяция выполнена для {file} -> {output_file}")
        except Exception as e:
            print(f"Ошибка обработки файла {file}: {e}")
    
    def interpolate_group(self, group_data, group_name):
        """
        Интерполяция изображений группы без обрезки, с унификацией размера пикселя.
        """
        group_output_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(group_output_dir, exist_ok=True)
        
        tiff_files_path = [os.path.join(self.input_dir, f"{file}") for file in group_data['file']]

        # Средний размер пикселя для группы в градусах широты и долготы
        avg_pixel_width, avg_pixel_height = self.calculate_average_pixel_size(group_data)

        # Интерполяция каждого файла без изменения экстента
        for file_path in tiff_files_path:
            if not os.path.exists(file_path):
                print(f'файла не существует')
                continue
            
            output_path = os.path.join(group_output_dir, f"{os.path.basename(file_path).replace('.tif', '')}_{group_name}_interpolated.tif")
            self.interpolate_file(avg_pixel_deg_size=(avg_pixel_width, avg_pixel_height), 
                                  file=file_path, 
                                  output_file=output_path)

    def process(self, geo_data_path, groups: dict):
        data = pd.read_csv(geo_data_path)
        
        # Обработка всех групп
        for group_name, group_images_names in groups.items():
            print(f"Обработка группы: {group_name}")
            group_data = data[data['file'].isin(group_images_names)]
            self.interpolate_group(group_data, group_name)