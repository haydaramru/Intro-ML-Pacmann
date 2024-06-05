import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
data = pd.read_csv("../data/mtcars.csv")
data.head()

# Drop model mobil
data = data.drop(columns = ["model"])
data.head()

# Memprediksi mpg (miles/gallon) -- ukuran konsumsi bahan bakar
# Eksplorasi
plt.figure(figsize = (15, 8))
sns.heatmap(data.corr(),
            annot = True)
plt.show()

''' ===== Prepare Data ===== '''
# Split input-output
# Buat input & output
def split_input_output(data, target_column):
    X = data.drop(columns = target_column)
    y = data[target_column]

    return X, y

X, y = split_input_output(data = data,
                          target_column = "mpg")
X.head()
y.head()

# Split train & test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.2,
                                                    random_state = 123)

X_train.head()
y_train.head()

''' ===== Melakukan Linear Regression ===== '''
# Menentukan baseline model
baseline_pred = np.mean(y_train)
print(baseline_pred)

from sklearn.dummy import DummyRegressor

# Create object
baseline_model = DummyRegressor(strategy='mean')

# Fit object
baseline_model.fit(X_train, y_train)

y_train_pred = baseline_model.predict(X_train)
y_train_pred

from sklearn.metrics import mean_squared_error

mse_baseline_train = mean_squared_error(y_true = y_train,
                                        y_pred = y_train_pred)
print(mse_baseline_train)

# Lakukan cross validation
from sklearn.model_selection import cross_val_score

scores_baseline = cross_val_score(estimator = baseline_model,
                                  X = X_train,
                                  y = y_train,
                                  cv = 5,
                                  scoring = 'neg_mean_squared_error')

mse_baseline_cv = -np.mean(scores_baseline)
mse_baseline_cv

# Buat Objek Linear Regression
from sklearn.linear_model import LinearRegression

# Buat objek
lr = LinearRegression()

# Lakukan fitting
lr.fit(X_train, y_train)

# Evaluasi model
# Predict y_train
y_train_pred = lr.predict(X_train)

# Cari MSE di data train
mse_lr_train = mean_squared_error(y_true = y_train,
                                  y_pred = y_train_pred)
print(mse_lr_train)

# Lakukan cross validation
scores_lr = cross_val_score(estimator = lr,
                            X = X_train,
                            y = y_train,
                            cv = 5,
                            scoring = "neg_mean_squared_error")

mse_lr_cv = -np.mean(scores_lr)
mse_lr_cv

# Tentukan model terbaik
model_summary = pd.DataFrame({"Model Name": ['Baseline', 'LinearRegression'],
                              "Model": [baseline_model, lr],
                              "MSE Train": [mse_baseline_train, mse_lr_train],
                              "MSE CV": [mse_baseline_cv, mse_lr_cv]})
model_summary

# Test Performa model terbaik di data Test
# Cek test scores
y_pred_test = lr.predict(X_test)

# Cari MSE data test
test_score = mean_squared_error(y_true = y_test,
                                y_pred = y_pred_test)
test_score

# Ekstrak model parameter
coef_ = lr.coef_
intercept_ = lr.intercept_
lr_params = np.append(coef_, intercept_)

lr_params = pd.DataFrame(lr_params,
                         index = list(X_train.columns) + ["constant"],
                         columns = ["coefficient"])
lr_params

# buat dalam bentuk fungsi
def fit_model(estimator, X_train, y_train):
    """Fungsi untuk fitting model"""
    # 1. Fitting model
    estimator.fit(X_train, y_train)

    # 2. Cari evaluasi di data train & valid
    y_pred_train = estimator.predict(X_train)
    train_score = mean_squared_error(y_true = y_train,
                                     y_pred = y_pred_train)

    valid_scores = cross_val_score(estimator = estimator,
                                   X = X_train,
                                   y = y_train,
                                   cv = 5,
                                   scoring = 'neg_mean_squared_error')
    cv_score = -np.mean(valid_scores)

    # 3. Ekstrak coefficient
    coef_ = estimator.coef_
    intercept_ = estimator.intercept_
    estimator_params = np.append(coef_, intercept_)

    estimator_params_df = pd.DataFrame(estimator_params,
                                       index = list(X_train.columns) + ["constant"],
                                       columns = ["coefficient"])

    return estimator, train_score, cv_score, estimator_params_df

lr, train_score, cv_score, lr_params_df = fit_model(estimator = LinearRegression(),
                                                    X_train = X_train,
                                                    y_train = y_train)
print(f"train score: {train_score:.3f}, cv score: {cv_score:.3f}")

''' ===== Melakukan Best Selection ===== '''
from itertools import combinations

column_list = list(X_train.columns)
n_column = len(column_list)

column_list

train_column_list = []

for i in range(n_column):
    list_of_combination = combinations(column_list, i)
    for combi in list_of_combination:
        train_column_list.append(list(combi))

# tambahkan seluruh kolom
train_column_list.append(column_list)
len(train_column_list)

idx = 95    # ambil salah satu kombinasi model
train_list_idx = train_column_list[idx]
train_list_idx

# Filter Data
X_train_idx = X_train[train_list_idx]
X_train_idx.head()

# Lakukan modeling
_, train_idx, cv_idx, _ = fit_model(estimator = LinearRegression(),
                                    X_train = X_train_idx,
                                    y_train = y_train)
print(f"train score: {train_idx:.3f}, cv score: {cv_idx:.3f}")

idx = 520   # ambil kombinasi lain
train_list_idx = train_column_list[idx]
train_list_idx

# Filter Data
X_train_idx = X_train[train_list_idx]
X_train_idx.head()

# Lakukan modeling
_, train_idx, cv_idx, _ = fit_model(estimator = LinearRegression(),
                                    X_train = X_train_idx,
                                    y_train = y_train)
print(f"train score: {train_idx:.3f}, cv score: {cv_idx:.3f}")

# Cari semua train & validation scores
train_score = []
cv_score = []

for idx in range(len(train_column_list)):
    if idx != 0:
        # Filter data
        train_list_idx = train_column_list[idx]
        X_train_idx = X_train[train_list_idx]

        # Buat model
        _, train_idx, cv_idx, _ = fit_model(estimator = LinearRegression(),
                                            X_train = X_train_idx,
                                            y_train = y_train)

        # Simpan hasil
        train_score.append(train_idx)
        cv_score.append(cv_idx)
        
# Plot hasil
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))

ax.boxplot([train_score, cv_score])

ax.set_xticklabels(["TRAIN", "CV"])
ax.set_ylabel("MSE")
plt.show()

# Cari best di data validasi
best_score = np.min(cv_score)
best_idx = np.argmin(cv_score)

best_idx, best_score

# Best features
train_column_list[best_idx + 1]

# Find model
lr_best, train_best_score, \
        cv_best_score, lr_params_best = fit_model(estimator = LinearRegression(),
                                                  X_train = X_train[train_column_list[best_idx+1]],
                                                  y_train = y_train)

print('Train score :', train_best_score)
print('CV score    :', cv_best_score)

lr_params_best

''' ===== Melakukan Regularisasi Ridge ===== '''
# Import Ridge
from sklearn.linear_model import Ridge

# Buat objek
ridge = Ridge(alpha = 1.0)

# Lakukan fitting
ridge.fit(X = X_train,
          y = y_train)

# Buat prediksi di data train
y_pred_train = ridge.predict(X_train)
train_score = mean_squared_error(y_train, y_pred_train)
train_score

# Lakukan cross validation
scores = cross_val_score(estimator = ridge,
                         X = X_train,
                         y = y_train,
                         cv = 5,
                         scoring = "neg_mean_squared_error")
scores

cv_score = -np.mean(scores)
cv_score

# Tampilkan parameter
coef_ = ridge.coef_
intercept_ = ridge.intercept_
params = np.append(coef_, intercept_)

params_df = pd.DataFrame(params,
                         index = list(X_train.columns) + ["constant"],
                         columns = ["coefficient"])
params_df

alpha = 1.0
_, train_score, cv_score, ridge_param = fit_model(estimator = Ridge(alpha=alpha),
                                                  X_train = X_train,
                                                  y_train = y_train)
print(f"train score: {train_score:.3f}, cv score: {cv_score:.3f}")
ridge_param

alpha = 10.5
_, train_score, cv_score, ridge_param = fit_model(estimator = Ridge(alpha=alpha),
                                                  X_train = X_train,
                                                  y_train = y_train)
print(f"train score: {train_score:.3f}, cv score: {cv_score:.3f}")
ridge_param

alphas = [0.5, 1.0, 2.5, 5.0, 7.5, 10.0,
          12.5, 15.0, 17.5, 30.0, 50.0]

mse_train_list = []
mse_cv_list = []
model_list = []

for alpha in alphas:
    model_i, train_score_i, \
        cv_score_i, model_param_i = fit_model(estimator = Ridge(alpha=alpha),
                                              X_train = X_train,
                                              y_train = y_train)
    mse_train_list.append(train_score_i)
    mse_cv_list.append(cv_score_i)
    model_list.append(model_param_i)

# Plot error
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

ax.plot(alphas, mse_train_list, c="r", marker=".", label="Train")
ax.plot(alphas, mse_cv_list, c="g", marker=".", label="CV")

ax.set_xlabel("alpha")
ax.set_ylabel("MSE")

plt.grid()
plt.legend()
plt.show()

# Best parameter adalah saat MSE di CV paling kecil
best_idx = np.argmin(mse_cv_list)
best_alpha = alphas[best_idx]
best_ridge_cv = mse_cv_list[best_idx]

best_alpha, best_ridge_cv

# Best model
best_param_ridge = model_list[best_idx]
best_param_ridge

# Buat summary plot
models = pd.concat(model_list, axis=1)
models.columns = alphas

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

for col in X_train.columns:
    ax.plot(alphas, models.loc[col], label=col, marker=".")

ax.set_xlabel("alpha")
ax.set_ylabel("coef")
plt.legend()
plt.grid()
plt.show()

# ===== cara lebih cepat =====
# Import grid search
from sklearn.model_selection import GridSearchCV

# Buat model & parameter model yang ingin divariasikan
ridge = Ridge()

param_space = {"alpha": alphas}
param_space

# Lakukan grid search dengan CV
cv_ridge = GridSearchCV(estimator = ridge,
                        param_grid = param_space,
                        scoring = "neg_mean_squared_error",
                        cv = 5)

# Fit searching
cv_ridge.fit(X = X_train,
             y = y_train)

cv_ridge.best_params_

# Buat objek baru
best_ridge = Ridge(alpha = cv_ridge.best_params_["alpha"])

# Fit model
best_ridge.fit(X = X_train,
               y = y_train)

best_ridge.coef_

''' ===== Melakukan Regularisasi Lasso ===== '''
# Import Lasso
from sklearn.linear_model import Lasso

# Buat objek
lasso = Lasso(alpha = 0.1)

# Lakukan fitting
lasso.fit(X = X_train,
          y = y_train)

# Buat prediksi di data train
y_pred_train = lasso.predict(X_train)
train_score = mean_squared_error(y_train, y_pred_train)
train_score

# Lakukan cross validation
scores = cross_val_score(estimator = lasso,
                         X = X_train,
                         y = y_train,
                         cv = 5,
                         scoring = "neg_mean_squared_error")
scores

cv_score = -np.mean(scores)
cv_score

# Tampilkan parameter
coef_ = lasso.coef_
intercept_ = lasso.intercept_
params = np.append(coef_, intercept_)

params_df = pd.DataFrame(params,
                         index = list(X_train.columns) + ["constant"],
                         columns = ["coefficient"])
params_df

alpha = 0.1
_, train_score, cv_score, lasso_param = fit_model(estimator = Lasso(alpha=alpha),
                                                  X_train = X_train,
                                                  y_train = y_train)

print(f"train score: {train_score:.3f}, cv score: {cv_score:.3f}")
lasso_param

alpha = 0.01
_, train_score, cv_score, lasso_param = fit_model(estimator = Lasso(alpha=alpha),
                                                  X_train = X_train,
                                                  y_train = y_train)

print(f"train score: {train_score:.3f}, cv score: {cv_score:.3f}")
lasso_param

alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 1.00,
          1.25, 1.50, 1.75, 3.00, 5.00]

mse_train_list = []
mse_cv_list = []
model_list = []

for alpha in alphas:
    model_i, train_score_i, \
        cv_score_i, model_param_i = fit_model(estimator = Lasso(alpha=alpha),
                                              X_train = X_train,
                                              y_train = y_train)

    mse_train_list.append(train_score_i)
    mse_cv_list.append(cv_score_i)
    model_list.append(model_param_i)

# Plot error
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

ax.plot(alphas, mse_train_list, c="r", marker=".", label="Train")
ax.plot(alphas, mse_cv_list, c="g", marker=".", label="CV")

ax.set_xlabel("alpha")
ax.set_ylabel("MSE")

plt.grid()
plt.legend()
plt.show()

# Best parameter adalah saat MSE di CV paling kecil
best_idx = np.argmin(mse_cv_list)
best_alpha = alphas[best_idx]
best_lasso_cv = mse_cv_list[best_idx]
best_alpha, best_lasso_cv

# Best model
best_param_lasso = model_list[best_idx]
best_param_lasso

# Buat summary plot
models = pd.concat(model_list, axis=1)
models.columns = alphas

# Plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

for col in X_train.columns:
    ax.plot(alphas, models.loc[col], label=col, marker=".")

ax.set_xlabel("alpha")
ax.set_ylabel("coef")
plt.legend()
plt.grid()
plt.show()

# ===== cara lebih cepat =====
# Import grid search
from sklearn.model_selection import GridSearchCV

# Buat model & parameter model yang ingin divariasikan
lasso = Lasso()

param_space = {"alpha": alphas}
param_space

# Lakukan grid search dengan CV
cv_lasso = GridSearchCV(estimator = lasso,
                        param_grid = param_space,
                        scoring = "neg_mean_squared_error",
                        cv = 5)

# Fit searching
cv_lasso.fit(X = X_train,
             y = y_train)

cv_lasso.best_params_

# Buat objek baru
best_lasso = Lasso(alpha = cv_lasso.best_params_["alpha"])

# Fit model
best_lasso.fit(X = X_train,
               y = y_train)

best_lasso.coef_

''' ===== Comparison ===== '''
# comparing the model parameter
best_params = pd.concat([lr_params_df,
                         lr_params_best,
                         best_param_ridge,
                         best_param_lasso],
                        axis = 1)
best_params.columns = ["OLS full features", "OLS best features", "Ridge", "Lasso"]
best_params

# memilih model terbaik
best_scores_df = pd.DataFrame({"Model": ["Baseline", "OLS full features", "OLS best features", "Ridge", "Lasso"],
                               "CV Scores": [mse_baseline_cv, mse_lr_cv, cv_best_score, best_ridge_cv, best_lasso_cv]})
best_scores_df

# Cari score
def mse_model(estimator, X_test, y_test):
    # Predict
    y_pred = estimator.predict(X_test)

    # Cari mse
    mse = mean_squared_error(y_test, y_pred)

    return mse

lr_params_best

mse_model(estimator = lr_best,
          X_test = X_test[lr_params_best.index[:-1]],
          y_test = y_test)
