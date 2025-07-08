# src/api_client.py
import requests
import os
from datetime import datetime
import logging
from typing import Dict

# Logging sozlash
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, api_url: str):
        """API bilan ishlash klassi"""
        self.api_url = api_url

    def send_result(self, full_image_path: str, cropped_image_path: str, plate_data: Dict) -> bool:
        """Natijalarni API orqali jo'natish"""
        try:
            with open(full_image_path, 'rb') as full_image, open(cropped_image_path, 'rb') as cropped_image:
                files = [
                    ('full_image', (os.path.basename(full_image_path), full_image, 'image/jpeg')),
                    ('cropped_image', (os.path.basename(cropped_image_path), cropped_image, 'image/jpeg'))
                ]
                data = {
                    'plate_text': plate_data.get('plate_text', ''),
                    'confidence': plate_data.get('confidence_avg', 0.0),
                    'is_valid': plate_data.get('is_valid_format', False),
                    'timestamp': datetime.now().isoformat()
                }
                response = requests.post(self.api_url, files=files, data=data)
                if response.status_code == 200:
                    logger.info(f"✅ Natija muvaffaqiyatli jo'natildi: {plate_data['plate_text']}")
                    return True
                else:
                    logger.error(f"❌ API xatosi: {response.status_code} - {response.text}")
                    return False
        except Exception as e:
            logger.error(f"❌ API jo'natishda xatolik: {str(e)}")
            return False