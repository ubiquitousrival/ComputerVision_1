import cv2
import numpy as np
import os

def read_binary_image(path):
    # Завантаження та перетворення у бінарну матрицю (0 та 1)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    return binary

def save_image(path, img_binary):
    # Конвертація назад у 0-255 для збереження
    cv2.imwrite(path, (img_binary * 255).astype(np.uint8))

def padding(image, pad_size):
    return np.pad(image, pad_size, mode='constant', constant_values=0)

# Реалізація операцій

def custom_erosion(image, kernel_size=3):
    rows, cols = image.shape
    pad_size = kernel_size // 2
    padded_image = padding(image, pad_size)
    output = np.zeros_like(image)
    kernel = np.ones((kernel_size, kernel_size))

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            # 1 тільки якщо всі пікселі під ядром = 1
            if np.array_equal(region, kernel):
                output[i, j] = 1
            else:
                output[i, j] = 0
    return output

def custom_dilation(image, kernel_size=3):
    rows, cols = image.shape
    pad_size = kernel_size // 2
    padded_image = padding(image, pad_size)
    output = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            # 1 якщо хоча б один піксель під ядром = 1
            if np.max(region) == 1:
                output[i, j] = 1
            else:
                output[i, j] = 0
    return output

def custom_opening(image, kernel_size=3):
    # Ерозія -> Дилатація
    return custom_dilation(custom_erosion(image, kernel_size), kernel_size)

def custom_closing(image, kernel_size=3):
    # Дилатація -> Ерозія
    return custom_erosion(custom_dilation(image, kernel_size), kernel_size)

def custom_boundary(image, kernel_size=3):
    # Оригінал - Ерозія
    return image - custom_erosion(image, kernel_size)

# Запуск 
if __name__ == "__main__":
    if not os.path.exists('results'):
        os.makedirs('results')

    input_path = "input.png"
    original = read_binary_image(input_path)

    save_image("results/1_erosion.png", custom_erosion(original))
    save_image("results/2_dilation.png", custom_dilation(original))
    save_image("results/3_opening.png", custom_opening(original))
    save_image("results/4_closing.png", custom_closing(original))
    save_image("results/5_boundary.png", custom_boundary(original))