## main.py
import json
import logging
from src.plate_detector import PlateDetector
from src.plate_reader import PlateReader
from src.api_client import APIClient

# Logging sozlash (faqat log fayliga yoziladi)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

# YOLO loglarini o'chirish
logging.getLogger('ultralytics').setLevel(logging.ERROR)

def load_config(config_path: str) -> dict:
    """Konfiguratsiyani o'qish"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Konfiguratsiyani o'qishda xatolik: {str(e)}")
        exit(1)

def main():
    """Asosiy funksiya"""
    # Konfiguratsiyani o'qish
    config = load_config('config/cameras.json')
    camera_config = config['cameras'][0]  # Birinchi kamerani olish
    api_url = "https://your-api-endpoint.com/api/upload"  # API manzilini o'zgartiring

    # Modellarni ishga tushirish
    detector = PlateDetector("models/best (6).pt", camera_config)
    reader = PlateReader("models/model.pt")
    api_client = APIClient(api_url)

    # Kamerani ishga tushirish
    detector.run(camera_config['source'], show_window=True)

    # Aniqlangan raqamlarni qayta ishlash
    for plate in detector.get_detected_plates():
        plate_result = reader.read_plate_from_image(plate['cropped_image'])
        if plate_result and plate_result['is_valid_format']:
            print(plate_result['plate_text'])  # Faqat raqam konsolga chiqadi
            # API orqali jo'natish
            api_client.send_result(plate['full_image'], plate['cropped_image'], plate_result)

if __name__ == "__main__":
    main()