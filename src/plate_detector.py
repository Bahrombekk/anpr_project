import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional

# Logging sozlash
logger = logging.getLogger(__name__)

class PlateDetector:
    def __init__(self, model_path: str, config: Dict, plate_reader=None):
        """Raqam aniqlash klassi"""
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logger.error(f"YOLO modelini yuklashda xatolik: {str(e)}")
            exit(1)
        
        self.cap = None
        self.lines = config['lines']
        self.detected_plates = []
        self.plate_reader = plate_reader  # PlateReader obyektini qo'shdik
        self.output_dir = "data/output"
        self.cropped_dir = os.path.join(self.output_dir, "cropped_plates")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cropped_dir, exist_ok=True)
        self.saved_objects = set()
        self.frame_object_ids = set()
        self.previous_centers = {}

    def start_source(self, source: str) -> bool:
        """Kamera yoki video faylni ishga tushirish"""
        self.cap = cv2.VideoCapture(source)
        if source.lower().endswith('.mov'):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))
        if not self.cap.isOpened():
            logger.error(f"Kamera/video ochilmadi: {source}")
            return False
        return True

    def is_near_or_crossed_line(self, cx: int, cy: int, line: Dict, threshold: int = 100) -> bool:
        """Nuqta chiziq yaqinida yoki kesib o'tganligini tekshirish"""
        x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
        if x2 == x1:
            return abs(cx - x1) < threshold
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        expected_y = slope * cx + intercept
        return abs(cy - expected_y) < threshold

    def is_between_lines(self, cx: int, cy: int, line2: Dict, line3: Dict) -> bool:
        """Nuqta ikki chiziq orasida ekanligini tekshirish"""
        min_x = min(line2['x1'], line2['x2'], line3['x1'], line3['x2'])
        max_x = max(line2['x1'], line2['x2'], line3['x1'], line3['x2'])
        min_y = min(line2['y1'], line2['y2'], line3['y1'], line3['y2'])
        max_y = max(line2['y1'], line2['y2'], line3['y1'], line3['y2'])
        return min_x <= cx <= max_x and min_y <= cy <= max_y

    def read_plate_and_print(self, cropped_image_path: str) -> Optional[Dict]:
        """Raqamni o'qish va konsolga chiqarish"""
        if self.plate_reader:
            plate_result = self.plate_reader.read_plate_from_image(cropped_image_path)
            if plate_result:
                if plate_result['is_valid_format']:
                    print(f"🚗 ANIQLANGAN RAQAM: {plate_result['plate_text']}")
                    logger.info(f"✅ Valid raqam: {plate_result['plate_text']}")
                else:
                    print(f"⚠️ NOTO'G'RI FORMAT: {plate_result['plate_text']}")
                    logger.warning(f"⚠️ Noto'g'ri format: {plate_result['plate_text']}")
                return plate_result
            else:
                print("❌ Raqam o'qib bo'lmadi")
                logger.error("❌ Raqam o'qib bo'lmadi")
        return None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """Kadrni qayta ishlash va raqamlarni aniqlash"""
        frame_with_lines = frame.copy()
        detected_plates = []

        # Chiziqlarni chizish
        for line_id, line in self.lines.items():
            color = (0, 255, 0) if line_id == 'entry_line' else (255, 0, 0)
            cv2.line(frame_with_lines, (line['x1'], line['y1']), (line['x2'], line['y2']), color, 2)

        # Raqam joylarini aniqlash
        results = self.model(frame, verbose=False)
        self.frame_object_ids.clear()
        current_centers = {}

        for i, box in enumerate(results[0].boxes):
            x, y, x1_obj, y1_obj = map(int, box.xyxy[0])
            center_x = (x + x1_obj) // 2
            center_y = (y + y1_obj) // 2

            # Raqam joyini belgilash
            cv2.rectangle(frame_with_lines, (x, y), (x1_obj, y1_obj), (255, 0, 0), 2)
            cv2.putText(frame_with_lines, f"Plate {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Obyekt ID berish
            object_id = None
            min_distance = float('inf')
            for prev_id, (prev_x, prev_y) in self.previous_centers.items():
                distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    object_id = prev_id

            if object_id is None:
                object_id = f"plate_{len(self.saved_objects)}_{center_x}_{center_y}"

            current_centers[object_id] = (center_x, center_y)
            self.frame_object_ids.add(object_id)

            # Chiziqdan o'tish va rasm saqlash
            if (self.is_near_or_crossed_line(center_x, center_y, self.lines['entry_line']) and
                    self.is_between_lines(center_x, center_y, self.lines['left_guard'], self.lines['right_guard']) and
                    object_id not in self.saved_objects):
                
                if 0 <= x < frame.shape[1] and 0 <= x1_obj < frame.shape[1] and 0 <= y < frame.shape[0] and 0 <= y1_obj < frame.shape[0]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    full_image_path = os.path.join(self.output_dir, f"full_frame_{timestamp}.jpg")
                    cropped_image_path = os.path.join(self.cropped_dir, f"plate_{timestamp}.jpg")
                    
                    # Raqam qismini kesib olish
                    plate_image = frame[max(0, y):min(y1_obj, frame.shape[0]), max(0, x):min(x1_obj, frame.shape[1])]
                    
                    if plate_image.size > 0:
                        # Rasmlarni saqlash
                        cv2.imwrite(full_image_path, frame_with_lines, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        cv2.imwrite(cropped_image_path, plate_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        logger.info(f"✅ Rasm saqlandi: {full_image_path}")
                        logger.info(f"✅ Kesib olingan raqam: {cropped_image_path}")
                        
                        # Raqamni o'qish va konsolga chiqarish
                        plate_result = self.read_plate_and_print(cropped_image_path)
                        
                        self.saved_objects.add(object_id)
                        
                        plate_data = {
                            "full_image": full_image_path,
                            "cropped_image": cropped_image_path,
                            "timestamp": timestamp,
                            "plate_result": plate_result
                        }
                        
                        detected_plates.append(plate_data)
                        self.detected_plates.append(plate_data)  # Ro'yxatga qo'shish
                        
                    else:
                        logger.warning(f"⚠️ Kesilgan rasm bo'sh: {timestamp}")

        self.previous_centers = current_centers.copy()
        return frame_with_lines, detected_plates

    def run(self, source: str, show_window: bool = True):
        """Asosiy ishga tushirish funksiyasi"""
        if not self.start_source(source):
            return

        logger.info("Dastur ishga tushdi. Chiqish uchun 'q' tugmasini bosing.")
        print("🚗 Avtomobil raqamlarini kuzatish boshlandi...")
        print("Chiqish uchun 'q' tugmasini bosing")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Kadr o'qib bo'lmadi!")
                break

            processed_frame, detected_plates = self.process_frame(frame)

            if show_window:
                # Kadrni 50% kichraytirish
                height, width = processed_frame.shape[:2]
                resized_frame = cv2.resize(processed_frame, (width // 3, height // 3))
                cv2.imshow('License Plate Detection', resized_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        if show_window:
            cv2.destroyAllWindows()

    def get_detected_plates(self) -> List:
        """Aniqlangan raqamlar ro'yxatini olish"""
        return self.detected_plates