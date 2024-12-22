import rasterio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Функция для изменения разрешения с сохранением пропорций
def resize_with_aspect(image, scale_factor):
    height, width = image.shape
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

# Загрузка GeoTIF изображения
with rasterio.open('../data/tifs/ridge_2.tif') as src:
    image = src.read(1)  # для одноканального изображения

# Снижение разрешения GeoTIF (например, до 25% от оригинального)
scale_factor = 0.25  # коэффициент уменьшения
image_resized = resize_with_aspect(image, scale_factor)

# Загрузка маски PNG и преобразование в одноканальный формат
mask = np.array(Image.open('../data/images/Ridge_2_medial_axis.png').convert('L'))

# Изменение размера маски под уменьшенное GeoTIF
resized_mask = cv2.resize(mask, (image_resized.shape[1], image_resized.shape[0]), interpolation=cv2.INTER_NEAREST)

# Создание RGBA маски (исправлено)
overlay = np.zeros((resized_mask.shape[0], resized_mask.shape[1], 4), dtype=np.float64)  # Трёхмерный массив
overlay[:, :, 0] = np.where(resized_mask > 0, 255, 0)  # Красный канал
overlay[:, :, 3] = np.where(resized_mask > 0, 128, 0)  # Альфа-канал (прозрачность)

# Создание визуализации
plt.figure(figsize=(12, 8))

# Отображение GeoTIF
plt.imshow(image_resized, cmap='gray')

# Отображение маски
plt.imshow(overlay, interpolation='none')
plt.title('Снимок с наложенной маской торосов')
plt.axis('off')

# Сохранение результата
plt.savefig('../data/processed_output/mask_on_tif/ridge_2.png', bbox_inches='tight', dpi=300)
plt.show()