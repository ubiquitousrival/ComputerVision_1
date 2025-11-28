import cv2
import numpy as np
import os

def apply_convolution(image, kernel):
    h, w = image.shape[:2]
    kh, kw = kernel.shape[:2]
    ph, pw = kh // 2, kw // 2
    
    padded = cv2.copyMakeBorder(image, ph, ph, pw, pw, cv2.BORDER_REFLECT)
    output = np.zeros_like(image, dtype=np.float32)
    
    if len(image.shape) == 3:
        for c in range(3):
            for y in range(h):
                for x in range(w):
                    output[y, x, c] = np.sum(padded[y:y+kh, x:x+kw, c] * kernel)
    else:
        for y in range(h):
            for x in range(w):
                output[y, x] = np.sum(padded[y:y+kh, x:x+kw] * kernel)
                
    return np.clip(output, 0, 255).astype(np.uint8)

def get_gaussian_kernel(size, sigma=1.0):
    k = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return k / np.sum(k)

# Головний блок
img = cv2.imread('input.jpg')
os.makedirs('results', exist_ok=True)

# 1. Зсув
M = np.float32([[1, 0, 10], [0, 1, 20]])
cv2.imwrite('results/1_shifted.jpg', cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))

# 2. Інверсія
cv2.imwrite('results/2_inverted.jpg', 255 - img)

# 3. Гаусс
cv2.imwrite('results/3_gaussian.jpg', apply_convolution(img, get_gaussian_kernel(11, 2)))

# 4. Motion Blur
k_motion = np.eye(9) / 9
cv2.imwrite('results/4_motion.jpg', apply_convolution(img, k_motion))

# 5. Різкість
k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
cv2.imwrite('results/5_sharpen.jpg', apply_convolution(img, k_sharp))

# 6. Собель
k_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
cv2.imwrite('results/6_sobel.jpg', apply_convolution(img, k_sobel))

# 7. Границі
k_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
cv2.imwrite('results/7_edge.jpg', apply_convolution(img, k_edge))

# 8. Кастомний (Emboss)
k_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
cv2.imwrite('results/8_custom.jpg', apply_convolution(img, k_emboss))