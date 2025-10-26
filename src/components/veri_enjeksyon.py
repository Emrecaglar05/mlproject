import os  # İşletim sistemiyle ilgili işlemler için (dosya yolu oluşturma vb.)
import sys  # Sistemle ilgili parametreler ve fonksiyonlar için
from src.exception import CustomException  # Kendi oluşturduğumuz özel hata sınıfı
from src.logger import logging  # Kendi oluşturduğumuz loglama modülü
import pandas as pd  # Veri analizi ve manipülasyonu için pandas kütüphanesi

from sklearn.model_selection import train_test_split  # Veri setini eğitim ve test olarak bölmek için
from dataclasses import dataclass  # Veri sınıfları oluşturmak için kullanılan bir decorator

from src.components.veri_dönüşümü import dataTransformation
from src.components.veri_dönüşümü import dataTransformationConfig

from src.components.model_egitimi import ModelTrainerConfig
from src.components.model_egitimi import ModelTrainer


# Veri enjeksiyonu adımı için gerekli dosya yollarını tutan bir konfigürasyon sınıfı
@dataclass
class data_enjeksyon_config:
    # os.path.join ile işletim sistemine uygun şekilde dosya yolları oluşturuluyor
    train_data_path: str=os.path.join('artifacts',"train.csv")  # Eğitim verisinin kaydedileceği yol
    test_data_path: str=os.path.join('artifacts',"test.csv")   # Test verisinin kaydedileceği yol
    raw_data_path: str=os.path.join('artifacts',"data.csv")    # Ham verinin (orijinal verinin) kaydedileceği yol


# Veri enjeksiyonu (Data Ingestion) işlemlerini yapan sınıf
class data_enjeksyon:
    def __init__(self):
        # Sınıf başlatıldığında, yukarıda tanımlanan konfigürasyon sınıfından bir nesne oluşturur.
        # Bu sayede dosya yollarına self.ingestion_config üzerinden erişilebilir.
        self.ingestion_config=data_enjeksyon_config()

    
    # Veri enjeksiyonu sürecini başlatan ana metot
    def initiate_data_enjeksyon(self):
        logging.info("Veri enjeksiyonu süreci başlatıldı") # Log dosyasına sürecin başladığına dair bilgi notu düşer
        try:
            # 'notebook/data/stud.csv' dosyasını okuyarak bir pandas DataFrame'i oluşturur
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Veri seti CSV dosyasından başarıyla okundu') # Log dosyasına bilgi notu düşer
            
            # Kayıt yapılacak 'artifacts' klasörünün var olduğundan emin olur. Eğer yoksa oluşturur.
            # os.path.dirname ile dosya yolundan klasör yolu elde edilir.
            # exist_ok=True, klasör zaten varsa hata vermesini engeller.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            
            # Okunan ham veriyi, konfigürasyonda belirtilen yola (artifacts/data.csv) kaydeder.
            # index=False, pandas'ın satır indexlerini CSV'ye yazmasını engeller.
            # header=True, sütun isimlerinin CSV'ye yazılmasını sağlar.
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Veri seti eğitim ve test olarak bölünmeye başlandı") # Log dosyasına bilgi notu düşer
            
            # Veri setini %80 eğitim, %20 test olacak şekilde böler.
            # test_size=0.2, verinin %20'sinin test seti olacağını belirtir.
            # random_state=42, bölme işleminin her seferinde aynı şekilde yapılmasını sağlar (tekrarlanabilirlik için önemlidir).
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            # Ayrılan eğitim setini konfigürasyondaki yola (artifacts/train.csv) kaydeder.
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            
            # Ayrılan test setini konfigürasyondaki yola (artifacts/test.csv) kaydeder.
            # *** NOT: Orijinal kodunuzda burada train_data_path yazıyordu, bu bir hataydı. 
            # Test setini yanlışlıkla train setinin üzerine yazardı. test_data_path olarak düzelttim. ***
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Veri enjeksiyonu ve bölme işlemi başarıyla tamamlandı") # Log dosyasına bilgi notu düşer
            
            # İşlem tamamlandıktan sonra oluşturulan eğitim, test ve ham veri dosyalarının yollarını döndürür.
            # Bu yollar, makine öğrenmesi boru hattının (pipeline) sonraki adımlarında kullanılacaktır.
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                # Genellikle raw_data_path'i döndürmeye gerek olmayabilir ama projenin yapısına göre kalabilir.
            )
        except Exception as e:
            # try bloğu içinde herhangi bir hata oluşursa, bu blok çalışır.
            # Oluşan hatayı kendi özel hata sınıfımız (CustomException) ile yakalayıp loglar.
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=data_enjeksyon()
    train_data,test_data=obj.initiate_data_enjeksyon()

    data_transformation=dataTransformation()
    
    # 1. Düzeltme: '_' yerine anlamlı bir değişken adı kullanıyoruz
    train_arr,test_arr,preprocessor_file_path=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    
    # 2. Düzeltme: Yakaladığımız 'preprocessor_file_path' değişkenini fonksiyona veriyoruz
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr,preprocessor_file_path))

    