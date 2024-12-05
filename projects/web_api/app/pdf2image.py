import io

import pymupdf  # PyMuPDF
import cv2
import numpy as np

def pdf_to_images(pdf_path, output_folder, start_page=0, end_page=None, dpi=300):
    # 打开PDF文件
    pdf_document = pymupdf.open(pdf_path)

    # 如果end_page未指定，则转换到最后一页
    if end_page is None:
        end_page = len(pdf_document)

    images = []

    for page_num in range(start_page, min(end_page, len(pdf_document))):
        page = pdf_document.load_page(page_num)  # 加载每一页
        pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))  # 将页面转换为图像，设置分辨率
        output_path = f"{output_folder}/page_{page_num + 1}.png"
        # pix.save(output_path)  # 保存图像

        # 将pymupdf的Pixmap转换为OpenCV的图像
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # 转换为灰度图像
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 降噪处理（使用中值滤波）  降噪会导致文字模糊，先调小
        img_gray = cv2.medianBlur(img_gray, 1)

        # 提高对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

        # 二值化处理
        # _, img_gray = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY)


        # 使用Otsu's 方法进行二值化
        _, img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 使用 cv2.imencode 将图像编码为 PNG 格式，并写入 BytesIO 流
        success, encoded_image = cv2.imencode('.png', img_gray)
        if not success:
            raise Exception("Failed to encode image")

        image_stream = io.BytesIO(encoded_image.tobytes())
        image_stream.seek(0)  # 重置指针到流的开头
        images.append(image_stream)

    return images

    # 示例调用
# pdf_to_images("/Users/lee/Downloads/倾斜文件/scan_20010104031321_18.pdf", "/Users/lee/Downloads/tempimg")
# pdf_to_images("/Users/lee/Downloads/scan_20010104031321_8.pdf", "/Users/lee/Downloads/tempimg")
# pdf_to_images("/Users/lee/Downloads/Scan_0001_2.pdf", "/Users/lee/Downloads/tempimg")

