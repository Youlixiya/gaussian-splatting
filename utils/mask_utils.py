import cv2
import numpy as np

def draw_mask(image, mask):
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image[~mask, :] //= 2
    image = image.astype(np.uint8)
    # cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=20)
    cv2.drawContours(image, contours, -1, (70, 130, 200), thickness=3)
    # return image