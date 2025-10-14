import sys     # hata bilgilerini almak için kullanılan kütüphane.
from logger import logging  # logger.py içindeki logging'i kullan
def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()              # şuanda oluşan hatanın iz bilgilerini alır.
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error accured in python script name [{0}] line number [{1}] error message[{2}]".format(    
        file_name,exc_tb.tb_lineno,str(error))  # Hata mesajını dosya adı, satır numarası ve hata mesajıyla birleştirir 
        
    return error_message
 
  
# 🔹 Özel hata sınıfı (Custom Exception)
class CustomException(Exception):   # "Exception" sınıfından miras alır (hata sınıfı oluşturmak için)
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)   # Üst sınıfın (Exception) init metodunu çağırır
       
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Kendi hata mesajını detaylı hale getirir (hangi dosya, satır, hata mesajı)

    def __str__(self):   # Hata yazdırıldığında ekrana nasıl görüneceğini belirler
        return self.error_message   
    

if __name__=="__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Divide by Zero")   # Hata mesajını ve traceback'i logla
        raise CustomException(e, sys)
