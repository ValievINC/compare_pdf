import sys

import fitz
from PIL import Image
import numpy as np
import cv2


def convert_pdf_to_images(pdf_path):
    """
    Конвертация PDF в img
    """
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        image_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(image_data)
    return images


def preprocess_image(image):
    """
    Препроцессинг для нормальной работы библиотек
    """
    image_gray = image.convert("L")
    image_array = np.array(image_gray)
    return image_array


def compare_images(image1, image2):
    """
    Сравниваем изображения по графикам и контурам. Получаем информацию в процентах
    """
    img1_processed = preprocess_image(image1)
    img2_processed = preprocess_image(image2)

    graphs_standard = cv2.calcHist([img1_processed], [0], None, [256], [0, 256])
    graphs_students = cv2.calcHist([img2_processed], [0], None, [256], [0, 256])

    graphs_standard_normalize = cv2.normalize(graphs_standard, graphs_standard).flatten()
    graphs_student_normalize = cv2.normalize(graphs_students, graphs_students).flatten()

    similarity = cv2.compareHist(graphs_standard_normalize, graphs_student_normalize, cv2.HISTCMP_BHATTACHARYYA)

    similarity_percentage = (1 - similarity) * 100

    return similarity_percentage


standard = convert_pdf_to_images('etalon.pdf')
student = convert_pdf_to_images('report (10).pdf')


total_similarity = 0
num_pages = min(len(standard), len(student))

for i in range(num_pages):
    similarity = compare_images(standard[i], student[i])
    total_similarity += similarity

average_similarity = total_similarity / 2

print(f"Страницы сходятся на {average_similarity:.2f}%", file=sys.stderr)
