import cv2
import numpy as np
import os

def apply_convolution(image, kernel, bias=0):
    h, w = image.shape[:2]
    kh, kw = kernel.shape[:2]
    ph, pw = kh // 2, kw // 2
    
    padded = cv2.copyMakeBorder(image, ph, ph, pw, pw, cv2.BORDER_REFLECT)
    output = np.zeros_like(image, dtype=np.float32)
    
    channels = 3 if len(image.shape) == 3 else 1
    
    if channels == 3:
        for c in range(3):
            for y in range(h):
                for x in range(w):
                    output[y, x, c] = np.sum(padded[y:y+kh, x:x+kw, c] * kernel) + bias
    else:
        for y in range(h):
            for x in range(w):
                output[y, x] = np.sum(padded[y:y+kh, x:x+kw] * kernel) + bias
                
    return np.clip(output, 0, 255).astype(np.uint8)

def get_gaussian_kernel(size, sigma=1.0):
    k = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)), (size, size))
    return k / np.sum(k)

img = cv2.imread('input.jpg')
os.makedirs('results', exist_ok=True)

# 1. Shift (Convolutional implementation)
shift_x, shift_y = 10, 20
kh, kw = 2 * abs(shift_y) + 1, 2 * abs(shift_x) + 1
k_shift = np.zeros((kh, kw))
k_shift[0, 0] = 1
cv2.imwrite('results/1_shifted.jpg', apply_convolution(img, k_shift))

# 2. Inversion (Convolutional implementation: y = -1*x + 255)
cv2.imwrite('results/2_inverted.jpg', apply_convolution(img, np.array([[-1]]), bias=255))

# 3. Gaussian Blur
cv2.imwrite('results/3_gaussian.jpg', apply_convolution(img, get_gaussian_kernel(11, 2)))

# 4. Motion Blur
cv2.imwrite('results/4_motion.jpg', apply_convolution(img, np.eye(9) / 9))

# 5. Sharpening
k_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
cv2.imwrite('results/5_sharpen.jpg', apply_convolution(img, k_sharp))

# 6. Sobel Filter
k_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
cv2.imwrite('results/6_sobel.jpg', apply_convolution(img, k_sobel))

# 7. Edge Detection
k_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
cv2.imwrite('results/7_edge.jpg', apply_convolution(img, k_edge))

# 8. Custom Filter (Emboss)
k_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
cv2.imwrite('results/8_custom.jpg', apply_convolution(img, k_emboss))