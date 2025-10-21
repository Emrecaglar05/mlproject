import logging  # Python'un log sistemi
import os       # Dosya ve klasör işlemleri için
from datetime import datetime  # Tarih ve saat almak için


LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_H_%M,%S')}.log" # Log dosyası adını oluştur (örn: 10_14_2025_H_10_30,45.log)

# Log klasörünün yolunu oluştur
log_dir = os.path.join(os.getcwd(), "logs")  # Mevcut dizin içinde "logs" klasörü
os.makedirs(log_dir, exist_ok=True)         # Klasör yoksa oluştur, varsa hata verme


LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)  # Log dosyasının tam yolunu oluştur

# Logging ayarları
logging.basicConfig(
    filename=LOG_FILE_PATH,   # Logların yazılacağı dosya
    level=logging.INFO,      # INFO ve üzeri log seviyeleri kaydedilir
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log formatı
)

if __name__=="__main__":

    logging.info("İlk log")  # INFO seviyesinde mesaj kaydedildi
        

   
