import json
import logging
import os
from src.plate_detector import PlateDetector
from src.plate_reader import PlateReader
from src.api_client import APIClient

# Papkalarni yaratish
os.makedirs('data/logs', exist_ok=True)
os.makedirs('data/output', exist_ok=True)

# Logging sozlash (faqat log fayliga yoziladi)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/app.log'),
        logging.StreamHandler()  # Konsolga ham chiqarish uchun
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
    print("🚀 Avtomobil raqamlarini aniqlash tizimi ishga tushmoqda...")
    
    # Konfiguratsiyani o'qish
    config = load_config('config/cameras.json')
    camera_config = config['cameras'][0]  # Birinchi kamerani olish
    
    # Model fayllarini tekshirish
    detector_model = "models/best (6).pt"
    reader_model = "models/model.pt"
    
    if not os.path.exists(detector_model):
        logger.error(f"❌ Detector modeli topilmadi: {detector_model}")
        exit(1)
    
    if not os.path.exists(reader_model):
        logger.error(f"❌ Reader modeli topilmadi: {reader_model}")
        exit(1)
    
    # Video faylini tekshirish
    if not os.path.exists(camera_config['source']):
        logger.error(f"❌ Video fayl topilmadi: {camera_config['source']}")
        exit(1)
    
    try:
        # PlateReader obyektini yaratish
        print("📖 Raqam o'quvchi modelini yuklash...")
        reader = PlateReader(reader_model)
        
        # PlateDetector obyektini yaratish va PlateReader ni berish
        print("🔍 Raqam aniqlash modelini yuklash...")
        detector = PlateDetector(detector_model, camera_config, plate_reader=reader)
        
        # API client (ixtiyoriy)
        api_url = "https://your-api-endpoint.com/api/upload"  # API manzilini o'zgartiring
        api_client = APIClient(api_url)
        
        print("✅ Barcha modellar yuklandi!")
        print("🎥 Video qayta ishlanmoqda...")
        print("=" * 50)
        
        # Kamerani ishga tushirish
        detector.run(camera_config['source'], show_window=True)
        
        print("=" * 50)
        print(f"📊 Jami aniqlangan raqamlar: {len(detector.get_detected_plates())}")
        
        # Oxirida barcha natijalarni ko'rsatish
        for i, plate in enumerate(detector.get_detected_plates(), 1):
            print(f"{i}. {plate['timestamp']}: ", end="")
            if plate['plate_result']:
                print(f"{plate['plate_result']['plate_text']} (Ishonch: {plate['plate_result']['confidence_avg']:.2f})")
            else:
                print("O'qib bo'lmadi")
        
    except Exception as e:
        logger.error(f"❌ Xatolik yuz berdi: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()