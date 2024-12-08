import cv2
import numpy as np


def apply_SOBEL_filter(image):

    # Aplicar el filtro de Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular la magnitud del gradiente
    magnitud_gradiente = np.sqrt(sobel_x**2 + sobel_y**2)

    # Normalizar la magnitud del gradiente para mostrarla
    magnitud_gradiente = cv2.normalize(magnitud_gradiente, None, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return magnitud_gradiente
