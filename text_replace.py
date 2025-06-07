import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytesseract
from googletrans import Translator
import os


def detect_chinese_text(image):
    data = pytesseract.image_to_data(image, lang='chi_sim', output_type=pytesseract.Output.DICT)
    results = []
    for i, text in enumerate(data['text']):
        text = text.strip()
        conf = int(data['conf'][i]) if data['conf'][i] != '-1' else -1
        if conf > 60 and any('\u4e00' <= c <= '\u9fff' for c in text):
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            results.append({'text': text, 'box': (x, y, w, h)})
    return results


def translate_text(text):
    translator = Translator()
    try:
        translated = translator.translate(text, src='zh-cn', dest='en')
        return translated.text
    except Exception:
        return text


def create_mask(shape, boxes):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def inpaint_image(image, mask):
    return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def overlay_text(image, items):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for box, text in items:
        x, y, w, h = box
        draw.text((x, y), text, fill=(0, 0, 0), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def process_image(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(input_path)
    items = detect_chinese_text(image)
    boxes = [item['box'] for item in items]
    mask = create_mask(image.shape, boxes)
    cleaned = inpaint_image(image, mask)
    translated_items = []
    for item in items:
        en_text = translate_text(item['text'])
        translated_items.append((item['box'], en_text))
    result = overlay_text(cleaned, translated_items)
    cv2.imwrite(output_path, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replace Chinese text in images with English translation.')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    args = parser.parse_args()
    process_image(args.input, args.output)
