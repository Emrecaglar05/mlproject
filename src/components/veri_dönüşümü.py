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
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # ✅ Dosya uzantısı düzeltildi


class dataTransformation:
    def __init__(self):
        # Sınıf başlatıldığında (initialize) çağrılır
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
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # 1. Adım: Eksik verileri en sık görülen değerle doldur
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore'))  # 2. Adım: Kategorik verileri One-Hot Encoding ile sayısallaştır
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # İki pipeline'ı birleştirme
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''Veri dönüşüm sürecini başlatır'''
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train ve test verileri okundu (Read train and test data completed)")

            logging.info("Ön işleyici (preprocessing) nesnesi alınıyor...")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"  # ✅ Küçük harf düzeltmesi

            # ---- Eğitim verisini ayır ----
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # ---- Test verisini ayır ----
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Ön işleme (preprocessing) eğitim ve test verisine uygulanıyor...")

            # Dönüşüm işlemleri
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Ön işleme tamamlandı.")

            # İşlenmiş verileri birleştir
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # ✅ save_object fonksiyonundaki parametre düzeltildi
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Veri dönüşümü sırasında hata: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = dataTransformation()
    print("Veri dönüşüm nesnesi oluşturuldu ✅")

    # Güncellenmiş CSV yolu
    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"  # Eğer test verisi ayrı değilse aynı dosyayı kullanabilirsin

    # Veri dönüşümünü başlat
    train_arr, test_arr, preprocessor_path = obj.initiate_data_transformation(train_path, test_path)

    print("Veri dönüşümü tamamlandı ✅")
    print(f"Ön işleyici kaydedildi: {preprocessor_path}")
    print(f"Eğitim verisi boyutu: {train_arr.shape}")
    print(f"Test verisi boyutu: {test_arr.shape}")
