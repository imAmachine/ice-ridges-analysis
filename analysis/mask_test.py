import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

SOURCE_TIF_PATH = './data/tifs/'
MASKS_TIF_PATH = './data/images'

UNIFIED_TIF_PATH = './data/processed_output/unified_analyze/images'
OUTPUT_TIF_ON_MASK_PATH = './data/processed_output/mask_on_tif'

FILENAME = ''

# Функция для изменения разрешения с фиксированной шириной (2048 пикселей)
def resize_to_width(image, target_width):
    height, width = image.shape
    scale_factor = target_width / width
    new_height = int(height * scale_factor)
    resized_image = cv2.resize(image, (target_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

TARGET_WIDTH = 2048

for i in range(1, 11):
    # Загрузка GeoTIFF
    with rasterio.open(os.path.join(SOURCE_TIF_PATH, f'ridge_{i}.tif')) as src:
        image = src.read(1)

    # Масштабирование GeoTIFF до 2048 пикселей по ширине
    image_resized = resize_to_width(image, TARGET_WIDTH)

    # Загрузка маски и преобразование
    mask = np.array(Image.open(os.path.join(MASKS_TIF_PATH, f'Ridge_{i}_medial_axis.png')).convert('L'))
    mask_resized = cv2.resize(mask, (image_resized.shape[1], image_resized.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Создание RGBA-маски
    overlay = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 4), dtype=np.uint8)
    overlay[:, :, 0] = np.where(mask_resized > 0, 255, 0)  # Красный канал
    overlay[:, :, 3] = np.where(mask_resized > 0, 128, 0)  # Альфа-канал

    # Визуализация и сохранение
    plt.figure(figsize=(12, 8))
    plt.imshow(image_resized, cmap='gray')
    plt.imshow(overlay, interpolation='none')
    plt.title('Снимок с наложенной маской торосов')
    plt.axis('off')

    output_path = os.path.join(OUTPUT_TIF_ON_MASK_PATH, f'ridge_{i}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
