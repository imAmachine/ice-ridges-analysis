import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, morphology
from noise import pnoise2



### Просто для интереса добавил
class PerlinNoise:
    def __init__(self, fd):
        self.scale = 50.0
        self.octaves = 4 if fd > 1.6 else 2
        self.persistence = 0.5 if fd > 1.5 else 0.7
        self.lacunarity = 2.0 if fd > 1.5 else 1.5
    
    def generate_perlin_noise(self, width, height):
        """Генерация шума Перлина на основе фрактального анализа"""
        noise_array = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                noise_array[i][j] = pnoise2(j / self.scale, 
                                            i / self.scale, 
                                            octaves=self.octaves, 
                                            persistence=self.persistence, 
                                            lacunarity=self.lacunarity, 
                                            repeatx=1024, 
                                            repeaty=1024, 
                                            base=42)
        return noise_array

    def threshold_noise(self, noise_image, threshold=0.0):
        """Пороговая обработка: выделение путей торосов"""
        return (noise_image > threshold).astype(np.uint8)

    def morphological_operations(self, binary_image):
        """Морфологические операции для очистки и улучшения изображения"""
        dilated = morphology.dilation(binary_image, morphology.disk(3))
        eroded = morphology.erosion(dilated, morphology.disk(3))
        return eroded

    def visualize(self, mask, noise_image, thresholded_image):
        """Визуализация оригинальной маски, шума и выделенных путей"""
        # Визуализируем промежуточные шаги
        plt.figure(figsize=(25, 10))
        
        # Оригинальная маска
        plt.subplot(1, 3, 1)
        plt.imshow(mask, cmap='gray')
        plt.title('Original Mask')
        
        # Пороговое изображение
        plt.subplot(1, 3, 2)
        plt.imshow(thresholded_image, cmap='gray')
        
        plt.subplot(1, 3, 3)
        plt.imshow(morphology.skeletonize(thresholded_image))
        
        plt.show()


