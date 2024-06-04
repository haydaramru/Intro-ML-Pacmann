import pandas as pd     # library pengolahan struktur data
import numpy as np      # library pengolahan angka

''' Importing Data to Python '''
# Load Data
data = pd.read_csv("../data/raw/credit_data.csv", sep = ";")
data.head()

# Output
# (Jumlah observasi, jumlah kolom/fitur)
data.shape

# cek data duplicate
duplicate_status = data.duplicated()
duplicate_status

# Cari jumlah data duplikatnya
# FALSE = 0 --> kalo tidak duplikat 
# TRUE = 1 --> kalo duplikat
# Kalau ada yang duplikat, maka jumlahnya > 0
duplicate_status.sum()

# Tidak ada yang di-drop karena tidak ada duplikat
data = data.drop_duplicates()

# Selalu sanity check!
# Periksa ulang jumlah observasi
data.shape

# Kita ingin membuat fungsi yang isi perintahnya sebagai berikut
data = pd.read_csv("../data/raw/credit_data.csv", sep=";")
print("Data asli            : ", data.shape, "- (#observasi, #kolom)")
data = data.drop_duplicates()
print("Data setelah di-drop : ", data.shape, "- (#observasi, #kolom)")

# ------- START HERE -------
# (filename) adalah argumen
# Argumen adalah sebuah variable. "
def importData(filename):       # Importing Data to Python
    """
    Fungsi untuk import data & hapus duplikat
    :param filename: <string> nama file input (format .csv)
    :return df: <pandas dataframe> sampel data
    """

    # read data
    df = pd.read_csv(filename, sep=";")
    print("Data asli            : ", df.shape, "- (#observasi, #kolom)")

    # drop duplicates
    df = df.drop_duplicates()
    print("Data setelah di-drop : ", df.shape, "- (#observasi, #kolom)")

    return df

# input
file_credit = "../data/raw/credit_data.csv"
# panggil fungsi
data = importData(filename = file_credit)

data.head()

''' Data Preprocessing '''
# buat data yang berisi data target
# pilih data dengan nama kolom `default.payment.next.month`, lalu namakan sebagai output_data
output_data = data["default.payment.next.month"]
output_data.head()

input_data = data.drop(["default.payment.next.month"], 
                       axis = 1)
input_data.head()

# isi perintah yang akan dimasukkan ke dalam fungsi
output_data = data["default.payment.next.month"]
input_data = data.drop("default.payment.next.month",
                       axis = 1)

# (data, output_column_name) adalah argumen
# Argumen adalah sebuah variable. 
# Jika fungsi tsb. diberi argumen data = credit_data, 
# maka semua variabel 'data' di dalam fungsi akan berubah menjadi credit_data
def extractInputOutput(data,
                       output_column_name,
                       column_to_drop):
    """
    Fungsi untuk memisahkan data input dan output
    :param data: <pandas dataframe> data seluruh sample
    :param output_column_name: <string> nama kolom output
    :return input_data: <pandas dataframe> data input
    :return output_data: <pandas series> data output
    """
    # drop data
    data = data.drop(columns = column_to_drop)
    output_data = data[output_column_name]
    input_data = data.drop(output_column_name,
                           axis = 1)
    
    return input_data, output_data

# Jangan sampai salah urutan dalam penempatan return
column_to_drop = ["Unnamed: 0"]
output_column_name = ["default.payment.next.month"]

X, y = extractInputOutput(data = data,
                          output_column_name = output_column_name,
                          column_to_drop = column_to_drop)

# Sanity check
X.head(2)
y.head(2)

for i in X.columns:
    print(i, ":", len(X[i].value_counts()))

# Check repayment status
repayment_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
for i in repayment_columns:
    print(X[i].value_counts(normalize=True))
    
# There are some classes of category that doesn't give information enough
# Based on these proportion below, we should grouping some classes:
# PAY_0 : class 4-8 into class 3
# PAY_2 : class 4-8 into class 3, class 1 into class 2
# PAY_3 : class 4-8 into class 3, class 1 into class 2
# PAY_4 : class 4-8 into class 3, class 1 into class 2
# PAY_5 : class 4-8 into class 3
# PAY_6 : class 4-8 into class 3
pay_list_to_3 = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
pay_list_to_2 = ["PAY_2", "PAY_3", "PAY_4"]

list_set_3 = [4.0, 5.0, 6.0, 7.0, 8.0]

# Convert to 3
for i in pay_list_to_3:
    for j in list_set_3:
        data.loc[data[i] == j, i] = 3.0

# Convert to 2
for i in pay_list_to_2:
    data.loc[data[i] == 1.0, i] = 2.0

for i in repayment_columns:
    print(data[i].value_counts(normalize=True))

# Education
# EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
data["EDUCATION"].value_counts(normalize=True)

# Terdapat kelas bernilai "0.0" yang tidak terdapat di Deskripsi Variabel
# kelas 4-6 pada kategori Education kurang dapat menjelaskan tingkat pendidikan dan sedikitnya informasi yang dapat diberikan
# kelas 0, 4, 5, dan 6 dijadikan satu kelas, yakni others atau kelas 4.0
edu_list_to_4 = [0.0, 5.0, 6.0]

for i in edu_list_to_4:
    data.loc[data['EDUCATION']== i, 'EDUCATION'] = 4.0
    
# redefine the input and output after change some values of class in repayment and education
X, y = extractInputOutput(data = data,
                          output_column_name = output_column_name,
                          column_to_drop = column_to_drop)

X.head(2)

# Import train-test splitting library dari sklearn (scikit learn)
from sklearn.model_selection import train_test_split

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.25,
                                                    random_state = 12)

# Sanity check hasil splitting
print(X_train.shape)
print(X_test.shape)

# Ratio
X_test.shape[0] / X.shape[0]
# Hasil 0.25 - sesuai dengan test_size kita

# Data Imputation
# Output: nama variabel, True/False.
# Jika True, maka ada data yang kosong
# Ada 2500-2700 data yang kosong
X_train.isnull().sum()

X_train.head()

X_train.columns

#_get_numeric_data() hanya akan mengambil column berisikan integer dan float
# hati-hati dengan data kategoric yang berbentuk integer!!
X_train_numerical = X_train._get_numeric_data() 

X_train_numerical.head()

# Buat kolom numerik
categorical_column = ["SEX", "EDUCATION", "MARRIAGE",
                      "PAY_0", "PAY_2", "PAY_3", "PAY_4",
                      "PAY_5", "PAY_6"]

numerical_column = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2",
                    "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4",
                    "PAY_AMT5", "PAY_AMT6"]

# Seleksi dataframe numerik
X_train_numerical = X_train[numerical_column]

X_train_numerical.head()

# Semua variabel numerical memiliki missing values
X_train_numerical.isnull().any()

from sklearn.impute import SimpleImputer

# namakan function SimpleImputer menjadi imputer, jangan lupa tanda kurung ()
# missing_values adalah tanda missing values dalam data.
#   - bisa NaN, bisa 999, bisa "KOSONG"
# Strategy median adalah strategy imputasi, 
# jika data kosong, diganti dengan median target
# Strategi lainnya adalah: mean
imputer = SimpleImputer(missing_values = np.nan,
                        strategy = "median")

# Isi perintah yang akan dibuat dalam fungsi
# Fit imputer
imputer.fit(X_train_numerical)
# Transform
imputed_data = imputer.transform(X_train_numerical)
X_train_numerical_imputed = pd.DataFrame(imputed_data)
X_train_numerical_imputed.columns = X_train_numerical.columns
X_train_numerical_imputed.index = X_train_numerical.index

X_train_numerical_imputed.isnull().any()

# Buat dalam bentuk fungsi
from sklearn.impute import SimpleImputer

def numericalImputation(data, numerical_column):
    """
    Fungsi untuk melakukan imputasi data numerik
    :param data: <pandas dataframe> sample data input
    :param numerical_column: <list> list kolom numerik data
    :return X_train_numerical: <pandas dataframe> data numerik
    :return imputer_numerical: numerical imputer method
    """
    # Filter data numerik
    numerical_data = data[numerical_column]

    # Buat imputer
    imputer_numerical = SimpleImputer(missing_values = np.nan,
                                      strategy = "median")
    imputer_numerical.fit(numerical_data)

    # Transform
    imputed_data = imputer_numerical.transform(numerical_data)
    numerical_data_imputed = pd.DataFrame(imputed_data)

    numerical_data_imputed.columns = numerical_column
    numerical_data_imputed.index = numerical_data.index

    return numerical_data_imputed, imputer_numerical

# Imputation Numeric
X_train_numerical, imputer_numerical = numericalImputation(data = X_train,
                                                           numerical_column = numerical_column)

X_train_numerical.isnull().any()

categorical_column

# Periksa lagi missing value
categorical_data = X_train[categorical_column]
categorical_data.isnull().sum()

# Kita isi kolom kategorik dengan "KOSONG"
categorical_data = X_train[categorical_column]
categorical_data = categorical_data.fillna(value="KOSONG")

categorical_data.isnull().sum()

# Buat dalam bentuk fungsi
def categoricalImputation(data, categorical_column):
    """
    Fungsi untuk melakukan imputasi data kategorik
    :param data: <pandas dataframe> sample data input
    :param categorical_column: <list> list kolom kategorikal data
    :return categorical_data: <pandas dataframe> data kategorikal
    """
    # seleksi data
    categorical_data = data[categorical_column]

    # lakukan imputasi
    categorical_data = categorical_data.fillna(value="KOSONG")

    return categorical_data

X_train_categorical = categoricalImputation(data = X_train,
                                            categorical_column = categorical_column)

X_train_categorical.isnull().sum()

# Preprocessing Categorical Variables
categorical_ohe = pd.get_dummies(X_train_categorical)
categorical_ohe.head(2)

# Buat menjadi fungsi
def extractCategorical(data, categorical_column):
    """
    Fungsi untuk ekstrak data kategorikal dengan One Hot Encoding
    :param data: <pandas dataframe> data sample
    :param categorical_column: <list> list kolom kategorik
    :return categorical_ohe: <pandas dataframe> data sample dengan ohe
    """
    data_categorical = categoricalImputation(data = data,
                                             categorical_column = categorical_column)
    categorical_ohe = pd.get_dummies(data_categorical)

    return categorical_ohe

X_train_categorical_ohe = extractCategorical(data = X_train,
                                             categorical_column = categorical_column)

X_train_categorical_ohe.head()

# Simpan kolom OHE untuk diimplementasikan dalam testing data
# Agar shape-nya konsisten
ohe_columns = X_train_categorical_ohe.columns
ohe_columns

X_train_concat = pd.concat([X_train_numerical,
                            X_train_categorical_ohe],
                           axis = 1)

X_train_concat.head()

X_train_concat.shape

X_train_concat.isnull().any()

# Standardizing Variables
from sklearn.preprocessing import StandardScaler

# Buat fungsi
def standardizerData(data):
    """
    Fungsi untuk melakukan standarisasi data
    :param data: <pandas dataframe> sampel data
    :return standardized_data: <pandas dataframe> sampel data standard
    :return standardizer: method untuk standardisasi data
    """
    data_columns = data.columns  # agar nama kolom tidak hilang
    data_index = data.index  # agar index tidak hilang

    # buat (fit) standardizer
    standardizer = StandardScaler()
    standardizer.fit(data)

    # transform data
    standardized_data_raw = standardizer.transform(data)
    standardized_data = pd.DataFrame(standardized_data_raw)
    standardized_data.columns = data_columns
    standardized_data.index = data_index

    return standardized_data, standardizer

X_train_clean, standardizer = standardizerData(data = X_train_concat)
X_train_clean.head()

''' Training Machine Learning '''
y_train.value_counts(normalize = True)
# baseline akurasi = 78%

# 1. Import Model
# Import dari sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 2. Fitting Model
# Model K nearest neighbor
knn = KNeighborsClassifier()
knn.fit(X_train_clean, y_train)

# Model Logistic Regression
logreg = LogisticRegression(random_state = 123)
logreg.fit(X_train_clean, y_train)

# Model Random Forest Classifier
random_forest = RandomForestClassifier(random_state = 123)
random_forest.fit(X_train_clean, y_train)

# Model Random Forest Classifier 1
# Mari kita ubah hyperparameter dari random forest --> n_estimator
# Maksud & tujuan akan dijelaskan pada kelas Random Forest
# Tambahkan n_estimator = 500
random_forest_1 = RandomForestClassifier(random_state = 123,
                                         n_estimators = 500)
random_forest_1.fit(X_train_clean, y_train)

# 3. Prediction
# Prediksi Logistic Regression
logreg.predict(X_train_clean)

predicted_logreg = pd.DataFrame(logreg.predict(X_train_clean))
predicted_logreg

predicted_knn = pd.DataFrame(knn.predict(X_train_clean))
predicted_knn.head()

predicted_rf = pd.DataFrame(random_forest.predict(X_train_clean))
predicted_rf.head()

predicted_rf_1 = pd.DataFrame(random_forest_1.predict(X_train_clean))
predicted_rf_1.head()

# 4. Cek performa model di data training
benchmark = y_train.value_counts(normalize=True)[0]
benchmark

# akurasi knn
knn.score(X_train_clean, y_train)

# akurasi logistic regression
logreg.score(X_train_clean, y_train)

# akurasi random forest
random_forest.score(X_train_clean, y_train)

# akurasi random forest 1
random_forest_1.score(X_train_clean, y_train)

# 5. Simpan model ke file pickle
import joblib
# Simpan model logreg ke dalam folder yang sama dengan notebook
# dengan nama logreg.pkl
joblib.dump(logreg, "logreg.pkl")

joblib.dump(knn, "knn.pkl")
joblib.dump(random_forest, "random_forest.pkl")
joblib.dump(random_forest_1, "random_forest_1.pkl")

# 6. Test Prediction
def extractTest(data,
                numerical_column, categorical_column, ohe_column,
                imputer_numerical, standardizer):
    """
    Fungsi untuk mengekstrak & membersihkan test data 
    :param data: <pandas dataframe> sampel data test
    :param numerical_column: <list> kolom numerik
    :param categorical_column: <list> kolom kategorik
    :param ohe_column: <list> kolom one-hot-encoding dari data kategorik
    :param imputer_numerical: <sklearn method> imputer data numerik
    :param standardizer: <sklearn method> standardizer data
    :return cleaned_data: <pandas dataframe> data final
    """
    # Filter data
    numerical_data = data[numerical_column]
    categorical_data = data[categorical_column]

    # Proses data numerik
    numerical_data = pd.DataFrame(imputer_numerical.transform(numerical_data))
    numerical_data.columns = numerical_column
    numerical_data.index = data.index

    # Proses data kategorik
    categorical_data = categorical_data.fillna(value="KOSONG")
    categorical_data.index = data.index
    categorical_data = pd.get_dummies(categorical_data)
    categorical_data.reindex(index = categorical_data.index, 
                             columns = ohe_column)

    # Gabungkan data
    concat_data = pd.concat([numerical_data, categorical_data],
                             axis = 1)
    cleaned_data = pd.DataFrame(standardizer.transform(concat_data))
    cleaned_data.columns = concat_data.columns

    return cleaned_data

def testPrediction(X_test, y_test, classifier, compute_score):
    """
    Fungsi untuk mendapatkan prediksi dari model
    :param X_test: <pandas dataframe> input
    :param y_test: <pandas series> output/target
    :param classifier: <sklearn method> model klasifikasi
    :param compute_score: <bool> True: menampilkan score, False: tidak
    :return test_predict: <list> hasil prediksi data input
    :return score: <float> akurasi model
    """
    if compute_score:
        score = classifier.score(X_test, y_test)
        print(f"Accuracy : {score:.4f}")

    test_predict = classifier.predict(X_test)

    return test_predict, score

X_test_clean = extractTest(data = X_test,
                           numerical_column = numerical_column,
                           categorical_column = categorical_column,
                           ohe_column = ohe_columns,
                           imputer_numerical = imputer_numerical,
                           standardizer = standardizer)

X_test_clean.shape

# Logistic Regression Performance
logreg_test_predict, score = testPrediction(X_test = X_test_clean,
                                            y_test = y_test,
                                            classifier = logreg,
                                            compute_score = True)

# K nearest neighbor Performance
knn_test_predict, score = testPrediction(X_test = X_test_clean,
                                         y_test = y_test,
                                         classifier = knn,
                                         compute_score = True)

# Random Forest Performance
rf_test_predict, score = testPrediction(X_test = X_test_clean,
                                        y_test = y_test,
                                        classifier = random_forest,
                                        compute_score = True)

# Random Forest 1 Performance
rf_1_test_predict, score = testPrediction(X_test = X_test_clean,
                                          y_test = y_test,
                                          classifier = random_forest_1,
                                          compute_score = True) 
