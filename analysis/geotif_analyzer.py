import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from osgeo import gdal
from itertools import combinations
from math import radians, sin, cos, atan2, sqrt, pi
gdal.DontUseExceptions()

class GeoTiffAnalyzer:
    SEMI_MAJOR_AXIS = 6378137.0
    SEMI_MINOR_AXIS = 6356752.3142

    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        if not os.path.exists(input_folder):
            os.mkdir(input_folder)
        
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

    def geo_coords_to_meters(self, lat):
        """Перевод градусов широты в метры с учетом эллипсоида WGS-84"""
        e2 = 1 - (np.power(self.SEMI_MINOR_AXIS, 2) / np.power(self.SEMI_MAJOR_AXIS, 2))  # Квадрат эксцентриситета
        sin_lat = np.sin(np.radians(lat))
        cos_lat = np.cos(np.radians(lat))
        nu = self.SEMI_MAJOR_AXIS / np.sqrt(1 - e2 * np.power(sin_lat, 2))  # Радиус нормали
        meters_per_lat = (np.pi / 180) * self.SEMI_MAJOR_AXIS * (1 - e2) / np.power(1 - e2 * np.power(sin_lat, 2), 1.5)
        meters_per_lon = (np.pi / 180) * nu * cos_lat
        return meters_per_lat, meters_per_lon

    def get_info_from_geotiff(self, file_path):
        dataset = gdal.Open(file_path)
        if not dataset:
            raise ValueError(f"Failed to open: {file_path}")
        
        gt = dataset.GetGeoTransform()
        lat_center = gt[3] + (dataset.RasterYSize * gt[5] / 2) # широта центра растра
        meters_per_lat, meters_per_lon = self.geo_coords_to_meters(lat_center)
        
        # Координаты углов изображения
        upper_left_lon = gt[0]  # Долгота верхнего левого угла
        upper_left_lat = gt[3]  # Широта верхнего левого угла
        pixel_size_x_deg = gt[1]  # Пиксель по X (долгота)
        pixel_size_y_deg = gt[5]  # Пиксель по Y (широта)
        
        lower_left_lat = upper_left_lat + (dataset.RasterYSize * pixel_size_y_deg)
        upper_right_lon = upper_left_lon + (dataset.RasterXSize * pixel_size_x_deg)
        
        pixel_size_x = abs(gt[1] * meters_per_lon)
        pixel_size_y = abs(gt[5] * meters_per_lat)
        pixel_area = pixel_size_x * pixel_size_y
        
        ground_resolution_x = abs(upper_right_lon - upper_left_lon) * meters_per_lon
        ground_resolution_y = abs(upper_left_lat - lower_left_lat) * meters_per_lat
        ground_area_m2 = ground_resolution_x * ground_resolution_y
        
        return {
            'file': os.path.basename(file_path),
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'origin_x': gt[0],
            'origin_y': gt[3],
            'pixel_size_x_deg': pixel_size_x_deg,
            'pixel_size_y_deg': pixel_size_y_deg,
            'pixel_size_x_m': pixel_size_x,
            'pixel_size_y_m': pixel_size_y,
            'pixel_area': pixel_area,
            'ground_resolution_x': ground_resolution_x,
            'ground_resolution_y': ground_resolution_y,
            'ground_area_m2': ground_area_m2
        }

    def _process_geotifs(self):
        geo_data = []
        for file in os.listdir(self.input_folder):
            if file.lower().endswith(('.tif', '.tiff')):
                file_path = os.path.join(self.input_folder, file)
                geo_data.append(self.get_info_from_geotiff(file_path))
        
        df = pd.DataFrame(geo_data)
        output_path = os.path.join(self.output_folder, 'geo_data.csv')
        df.to_csv(output_path, index=False)
        return df

    def get_distances(self, geo_data):
        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            return 2 * self.SEMI_MAJOR_AXIS * atan2(sqrt(a), sqrt(1-a))
        
        distances = []
        files = geo_data[['file', 'origin_x', 'origin_y']].values
        
        # Расчет расстояний между парами
        for (f1, x1, y1), (f2, x2, y2) in combinations(files, 2):
            distances.append({
                'file1': f1,
                'file2': f2,
                'distance_m': np.float64(haversine(y1, x1, y2, x2))
            })
        
        # Добавление нулевых расстояний для одинаковых файлов
        distances.extend([
            {'file1': f, 'file2': f, 'distance_m': 0.0} 
            for f in geo_data['file']
        ])
        
        df = pd.DataFrame(distances)
        output_path = os.path.join(self.output_folder, 'distances.csv')
        df.to_csv(output_path, index=False)
        return df

    def plot_distances(self, distances_df):
        output_path = os.path.join(self.output_folder, 'distance_matrix.png')
        
        files = sorted(set(distances_df['file1']).union(distances_df['file2']))
        matrix = pd.DataFrame(0.0, index=files, columns=files)
        
        for _, row in distances_df.iterrows():
            matrix.at[row['file1'], row['file2']] = float(row['distance_m'])
            matrix.at[row['file2'], row['file1']] = float(row['distance_m'])

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            cmap='coolwarm',
            cbar_kws={'label': 'Distance (m)'},
            linewidths=0.5
        )
        plt.title("Distance Matrix Between Images")
        plt.xlabel("Image")
        plt.ylabel("Image")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_resolutions(self, geo_data):
        output_path = os.path.join(self.output_folder, 'average_resolution.png')
        
        geo_data['avg_resolution'] = (geo_data['pixel_size_x_m'] + geo_data['pixel_size_y_m']) / 2
        plt.figure(figsize=(14, 8))
        sns.barplot(data=geo_data, x='file', y='avg_resolution', hue='file', dodge=False, palette='viridis')
        plt.title("Average Resolution per Image")
        plt.xlabel("Image")
        plt.ylabel("Average Resolution (m)")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def analyze(self):
        """Выполнить полный анализ"""
        geo_data = self._process_geotifs()
        distances = self.get_distances(geo_data)
        self.plot_distances(distances)
        self.plot_resolutions(geo_data)
