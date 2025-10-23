import sys  # Sistemle ilgili parametreler ve fonksiyonlar için (hata yönetimi için sys)
import os  # İşletim sistemiyle ilgili işlemler için (dosya yolları oluşturmak gibi)
import pandas as pd  # Veri analizi ve manipülasyonu için pandas kütüphanesi
from dataclasses import dataclass  # Veri sınıfları (data classes) oluşturmak için kolay bir yol sağlar
import numpy as np  # Sayısal hesaplamalar ve array'ler için NumPy kütüphanesi
from sklearn.compose import ColumnTransformer  # Farklı sütunlara farklı dönüşümler uygulamak için
from sklearn.impute import SimpleImputer  # Eksik verileri doldurmak (imputation) için
from sklearn.pipeline import Pipeline  # Dönüşüm adımlarını bir arada tutan bir boru hattı oluşturmak için
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Kategorik veriyi (OHE) ve sayısal veriyi (Scaler) dönüştürmek için

from src.utils import save_object
from src.exception import CustomException  # Kendi oluşturduğumuz özel hata sınıfı (varsayılıyor)
from src.logger import logging  # Kendi oluşturduğumuz loglama modülü (varsayılıyor)

@dataclass  # Bu dekoratör, __init__ gibi metotları otomatik oluşturan bir sınıf tanımlar
class dataTransformationConfig:
    # Bu sınıf, veri dönüşümü sürecinde kullanılacak dosya yolları gibi konfigürasyonları saklar
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pk1")  # Ön işleyici nesnesinin (pickle dosyası) kaydedileceği yol

class dataTransformation:
    def __init__(self):
        # Sınıf başlatıldığında (initialize) çağrılır
        # HATA DÜZELTİLDİ: Sınıfın kendisi değil, bir ÖRNEĞİ (instance) oluşturulmalı.
        self.data_transformation_config = dataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Bu metot, sayısal ve kategorik sütunlar için ön işleme (preprocessing) 
        adımlarını tanımlayan bir ColumnTransformer nesnesi oluşturur ve döndürür.
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]  # Sayısal özelliklerin listesi
            categorical_columns = [  # Kategorik özelliklerin listesi
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Sayısal sütunlar için boru hattı (pipeline)
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # 1. Adım: Eksik verileri medyan ile doldur
                    ("scaler", StandardScaler(with_mean=False))     # 2. Adım: Veriyi standartlaştır (ölçeklendir)
                ]
            )

            # Kategorik sütunlar için boru hattı (pipeline)
            # SYNTAX HATASI DÜZELTİLDİ: '1' ve '412' satırları kaldırıldı.
            # MANTIKSAL HATA DÜZELTİLDİ: OneHotEncoder çıktısı zaten seyrektir (sparse), 
            # tekrar StandardScaler uygulamak (with_mean=False olsa bile) genellikle gereksizdir ve kaldırıldı.
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # 1. Adım: Eksik verileri en sık görülen değerle doldur
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))  # 2. Adım: Kategorik verileri One-Hot Encoding ile sayısallaştır
                    # (handle_unknown='ignore', test verisinde eğitimde olmayan bir kategori gelirse hata vermemesini sağlar)
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

           
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns), # Sayısal sütunlara 'num_pipeline' uygula
                    ("cat_pipelines", cat_pipeline, categorical_columns) # Kategorik sütunlara 'cat_pipeline' uygula
                ]
            )

            return preprocessor  # 'preprocessor' nesnesini döndür

        except Exception as e:
            raise CustomException(e, sys)  # Hata oluşursa özel hata sınıfımızla yakala ve fırlat

    def initiate_data_transformation(self, train_path, test_path):
        # Bu metot, veri dönüşüm sürecini başlatır
        try:
            train_df = pd.read_csv(train_path)  # Eğitim verisini CSV dosyasından oku
            test_df = pd.read_csv(test_path)  # Test verisini CSV dosyasından oku

            logging.info("Train ve test verileri okundu (Read train and test data completed)")

            logging.info("Ön işleyici (preprocessing) nesnesi alınıyor...")
            # Yukarıdaki 'get_data_transformer_object' metodunu çağır
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_Score"  # Hedef değişkenin (tahmin edilecek sütun) adı
            # numerical_columns = ["writing_score", "reading_score"] # Bu satıra burada gerek yok, get_data_transformer_object içinde tanımlı

            # ---- EĞİTİM VERİSİ AYIRMA ----
            # Eğitim verisinde hedef sütunu (target) ve özellik sütunlarını (features) ayır
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # ---- TEST VERİSİ AYIRMA ----
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Ön işleme (preprocessing) eğitim ve test verisine uygulanıyor.")

            # EKSİK KISIM EKLENDİ: Dönüşüm işlemlerini uygulama
            
            # fit_transform: Eğitim verisi üzerinde hem öğrenir (imputer, scaler vb.) hem de dönüştürür
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # transform: Test verisine, SADECE eğitim verisinden öğrendiği parametrelerle dönüştürme uygular
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Ön işleme tamamlandı.")

            # İşlenmiş özellikleri (input_feature...arr) ve hedefi (target_feature...df) birleştirme
            # np.c_ (column stack) kullanarak array'leri yanyana birleştiriyoruz
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj_path=preprocessing_obj
            )

            # İşlenmiş array'leri ve preprocessor'ın yolunu döndür
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        
        except Exception as e:
            logging.error(f"Veri dönüşümü sırasında hata: {e}")
            raise CustomException(e, sys)