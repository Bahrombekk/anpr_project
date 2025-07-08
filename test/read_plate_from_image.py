import cv2
import numpy as np
from ultralytics import YOLO
import re
import logging
from typing import Optional, Dict

# Logging sozlash
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlateReader:
    def __init__(self, model_path: str = "models/model.pt"):
        """
        Avtomobil raqami o'quvchi klassi
        
        Args:
            model_path: YOLO model fayl yo'li
        """
        self.model_path = model_path
        self.model = self.load_model()
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        
        # Belgilar xaritasi
        self.uzbek_plate_mapping = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F',
            'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L',
            'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R',
            'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
            'Y': 'Y', 'Z': 'Z'
        }
        
        # O'zbekiston avtomobil raqam formatlari
        self.uzbek_plate_patterns = [
            r'^[0-9]{2}[A-Z]{1}[0-9]{3}[A-Z]{2}$',  # 01A123BC
            r'^[0-9]{2}[A-Z]{3}[0-9]{3}$',          # 01ABC123
            r'^[0-9]{2}[0-9]{3}[A-Z]{3}$'   # A123BC01
        ]

    def load_model(self) -> YOLO:
        """YOLO modelini yuklash"""
        try:
            model = YOLO(self.model_path)
            logger.info(f"✅ Model muvaffaqiyatli yuklandi: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"❌ Model yuklashda xatolik: {str(e)}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Rasmni oldindan qayta ishlash
        
        Args:
            image: Kiruvchi rasm
            
        Returns:
            Qayta ishlangan rasm
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        return denoised

    def filter_detections(self, boxes) -> list:
        """
        Aniqlangan belgilarni filtrlash
        
        Args:
            boxes: YOLO detection natijalari
            
        Returns:
            Filtrlangan belgilar ro'yxati
        """
        return [box for box in boxes if box.conf.item() >= self.confidence_threshold]

    def post_process_text(self, raw_chars: list) -> str:
        """
        Aniqlangan belgilarni raqam va harflarga konvertatsiya qilish
        
        Args:
            raw_chars: YOLO modelidan olingan xom sinf nomlari ro'yxati
            
        Returns:
            Qayta ishlangan raqam matni
        """
        processed_text = ""
        for char in raw_chars:
            processed_text += self.uzbek_plate_mapping.get(char, char)
        
        # Umumiy xatoliklarni to'g'rilash
        corrections = {'O': '0', 'I': '1', 'S': '5', 'G': '6'}
        corrected_text = ""
        for i, char in enumerate(processed_text):
            if self.should_be_number(i, len(processed_text)) and char in corrections:
                corrected_text += corrections[char]
            else:
                corrected_text += char
        
        return corrected_text

    def should_be_number(self, position: int, total_length: int) -> bool:
        """
        Berilgan pozitsiyada raqam bo'lishi kerakligini aniqlash
        
        Args:
            position: Belgi pozitsiyasi
            total_length: Umumiy uzunlik
            
        Returns:
            True agar raqam bo'lishi kerak bo'lsa
        """
        if total_length == 8:
            return position in [0, 1, 3, 4, 5]
        return False

    def validate_plate_format(self, plate_text: str) -> bool:
        """
        Raqam formatini validatsiya qilish
        
        Args:
            plate_text: Tekshirilayotgan raqam
            
        Returns:
            True agar format to'g'ri bo'lsa
        """
        for pattern in self.uzbek_plate_patterns:
            if re.match(pattern, plate_text):
                return True
        return False

    def read_plate_from_image(self, image_path: str) -> Optional[Dict]:
        """
        Rasm dan avtomobil raqamini o'qish
        
        Args:
            image_path: Rasm fayl yo'li
            
        Returns:
            Natija dictionary yoki None
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"❌ Rasm topilmadi: {image_path}")
                return None
            
            logger.info(f"📸 Rasm o'qildi: {image_path}")
            processed_image = self.preprocess_image(image)
            results = self.model(processed_image, conf=self.confidence_threshold, iou=self.iou_threshold)
            boxes = results[0].boxes

            if not boxes:
                logger.warning("❌ Belgilar aniqlanmadi.")
                return None

            filtered_boxes = self.filter_detections(boxes)
            if not filtered_boxes:
                logger.warning("❌ Yetarli ishonch darajasiga ega belgilar topilmadi.")
                return None

            characters = [
                {
                    'x_min': int(box.xyxy[0][0].item()),
                    'y_min': int(box.xyxy[0][1].item()),
                    'x_max': int(box.xyxy[0][2].item()),
                    'y_max': int(box.xyxy[0][3].item()),
                    'cls_id': int(box.cls.item()),
                    'confidence': box.conf.item()
                }
                for box in filtered_boxes
            ]

            characters.sort(key=lambda x: x['x_min'])
            label_map = results[0].names
            raw_chars = [label_map[char['cls_id']] for char in characters]
            plate_text = self.post_process_text(raw_chars)

            result = {
                'plate_text': plate_text,
                'raw_chars': raw_chars,
                'is_valid_format': self.validate_plate_format(plate_text),
                'confidence_avg': sum(char['confidence'] for char in characters) / len(characters),
                'character_count': len(characters)
            }

            if result['is_valid_format']:
                logger.info(f"✅ Aniqlangan raqam: {plate_text} (Ishonch: {result['confidence_avg']:.2f})")
            else:
                logger.warning(f"⚠️ Aniqlangan matn: {plate_text} (Format noto'g'ri)")
            
            return result

        except Exception as e:
            logger.error(f"❌ Xatolik yuz berdi: {str(e)}")
            return None

def main():
    """Asosiy test funksiyasi"""
    try:
        plate_reader = PlateReader("models/model.pt")
        image_path = "detected_plates/cropped/plate_20250708_153113_717061.jpg"
        result = plate_reader.read_plate_from_image(image_path)
        
        if result:
            print(f"\n📊 NATIJA:")
            print(f"   Raqam: {result['plate_text']}")
            print(f"   Xom belgilar: {result['raw_chars']}")
            print(f"   Format to'g'ri: {result['is_valid_format']}")
            print(f"   O'rtacha ishonch: {result['confidence_avg']:.2f}")
            print(f"   Belgilar soni: {result['character_count']}")

    except Exception as e:
        logger.error(f"❌ Asosiy funksiyada xatolik: {str(e)}")

if __name__ == "__main__":
    main()