import cv2
import numpy as np
from ultralytics import YOLO
import os
from datetime import datetime

class PlateDetector:
    def __init__(self):
        # Load YOLO model for license plate detection only
        try:
            self.model = YOLO("models/best (6).pt")
        except Exception as e:
            print(f"YOLO modelini yuklashda xatolik: {str(e)}")
            exit(1)

        # Initialize variables
        self.cap = None
        self.lines = {
            'line1': {'x1': 552, 'y1': 941, 'x2': 1996, 'y2': 931},
            'line2': {'x1': 493, 'y1': 555, 'x2': 597, 'y2': 1159},
            'line3': {'x1': 2024, 'y1': 562, 'x2': 1950, 'y2': 1117}
        }
        self.detected_plates = []
        self.output_dir = "detected_plates"
        self.cropped_dir = os.path.join(self.output_dir, "cropped")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cropped_dir, exist_ok=True)
        self.saved_objects = set()
        self.frame_object_ids = set()
        self.previous_centers = {}

    def start_camera(self, camera_index=0):
        """Kamerani ishga tushirish"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print("Kamera ochilmadi!")
            return False
        return True

    def start_video(self, video_path):
        """Video faylni ishga tushirish"""
        self.cap = cv2.VideoCapture(video_path)
        if video_path.lower().endswith('.mov'):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'))
        if not self.cap.isOpened():
            print("Video fayl ochilmadi!")
            return False
        return True

    def is_near_or_crossed_line(self, cx, cy, line, threshold=100):
        """Nuqta chiziq yaqinida yoki uni kesib o'tganligini tekshirish"""
        x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
        if x2 == x1:
            return abs(cx - x1) < threshold
        else:
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            expected_y = slope * cx + intercept
            return abs(cy - expected_y) < threshold

    def is_between_lines(self, cx, cy, line2, line3):
        """Nuqta ikki chiziq orasida ekanligini tekshirish"""
        x1_2, y1_2, x2_2, y2_2 = line2['x1'], line2['y1'], line2['x2'], line2['y2']
        x1_3, y1_3, x2_3, y2_3 = line3['x1'], line3['y1'], line3['x2'], line3['y2']
        min_x = min(x1_2, x2_2, x1_3, x2_3)
        max_x = max(x1_2, x2_2, x1_3, x2_3)
        min_y = min(y1_2, y2_2, y1_3, y2_3)
        max_y = max(y1_2, y2_2, y1_3, y2_3)
        return min_x <= cx <= max_x and min_y <= cy <= max_y

    def process_frame(self, frame):
        """Kadrni qayta ishlash va raqamlarni aniqlash"""
        frame_with_lines = frame.copy()
        
        # Chiziqlarni chizish
        for line_id, line in self.lines.items():
            color = (0, 255, 0) if line_id == 'line1' else (255, 0, 0)
            cv2.line(frame_with_lines, (line['x1'], line['y1']), (line['x2'], line['y2']), color, 2)

        # Raqam joylarini aniqlash
        results = self.model(frame)
        self.frame_object_ids.clear()
        current_centers = {}

        for i, box in enumerate(results[0].boxes):
            x, y, x1_obj, y1_obj = map(int, box.xyxy[0])
            center_x = (x + x1_obj) // 2
            center_y = (y + y1_obj) // 2

            print(f"🟩 Raqam joyi aniqlandi: x={x}, y={y}, x1={x1_obj}, y1={y1_obj}, center=({center_x}, {center_y})")

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
            if (self.is_near_or_crossed_line(center_x, center_y, self.lines['line1'], threshold=100) and
                self.is_between_lines(center_x, center_y, self.lines['line2'], self.lines['line3']) and
                object_id not in self.saved_objects):
                
                if 0 <= x < frame.shape[1] and 0 <= x1_obj < frame.shape[1] and 0 <= y < frame.shape[0] and 0 <= y1_obj < frame.shape[0]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    
                    # To'liq kadrni saqlash
                    full_image_path = os.path.join(self.output_dir, f"full_frame_{timestamp}.jpg")
                    cv2.imwrite(full_image_path, frame_with_lines, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    # Raqam joyini kesib olish va saqlash
                    plate_image = frame[max(0, y):min(y1_obj, frame.shape[0]), max(0, x):min(x1_obj, frame.shape[1])]
                    if plate_image.size > 0:
                        cropped_image_path = os.path.join(self.cropped_dir, f"plate_{timestamp}.jpg")
                        cv2.imwrite(cropped_image_path, plate_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        print(f"✅ Rasm saqlandi: {full_image_path}")
                        print(f"✅ Kesib olingan raqam: {cropped_image_path}")
                        
                        # Obyektni saqlanganlarga qo'shish
                        self.saved_objects.add(object_id)
                        
                        # Natijalarni saqlash
                        self.detected_plates.append({
                            "full_image": full_image_path,
                            "cropped_image": cropped_image_path,
                            "timestamp": timestamp
                        })
                    else:
                        print(f"⚠️ Kesilgan rasm bo'sh: {timestamp}")

        self.previous_centers = current_centers.copy()
        return frame_with_lines

    def run(self, source=0, show_window=True):
        """Asosiy ishga tushirish funksiyasi"""
        # Kamera yoki video faylni ishga tushirish
        if isinstance(source, int):
            if not self.start_camera(source):
                return
        else:
            if not self.start_video(source):
                return

        print("Dastur ishga tushdi. Chiqish uchun 'q' tugmasini bosing.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Kadr o'qib bo'lmadi!")
                break

            # Kadrni qayta ishlash
            processed_frame = self.process_frame(frame)

            # Oyna ko'rsatish (ixtiyoriy)
            if show_window:
                cv2.imshow('License Plate Detection', processed_frame)
                
                # 'q' tugmasi bosilganda chiqish
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Tozalash
        self.cap.release()
        if show_window:
            cv2.destroyAllWindows()

    def set_lines(self, lines):
        """Chiziqlarni o'rnatish"""
        self.lines = lines

    def get_detected_plates(self):
        """Aniqlangan raqamlar ro'yxatini olish"""
        return self.detected_plates

def main():
    detector = PlateDetector()
    
    # Video faylni ishga tushirish
    video_path = "videos/20241219_20241219175204_20241219175234_175204.mp4"
    detector.run(source=video_path, show_window=True)
    
    # Aniqlangan raqamlar ro'yxatini olish
    detected = detector.get_detected_plates()
    print(f"Jami {len(detected)} ta raqam aniqlandi va saqlandi.")

if __name__ == '__main__':
    main()