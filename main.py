import seaborn as sns
import  numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from random import sample
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from helpers.data_prep import
from helpers.eda import
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("Telco-Customer-Churn.csv")
dataframe.head()
df = dataframe.copy()

# Görev 1 : Keşifçi Veri Analizi
# 1.Adım : Genel resmi inceleyiniz
print(dataframe.shape)
print(dataframe.dtypes)
print(dataframe.head())
print(dataframe.isnull().sum())
print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# 2.Adım : Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th= 10, car_th= 20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#3. Adım: Numerik ve kategorik değişkenlerin analizini yapınız.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=False)
for col in cat_cols:
    cat_summary(df,col,True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = False)

for col in num_cols:
    num_summary(df, col, False)

#4. Adım: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene
# göre numerik değişkenlerin ortalaması)
df.dtypes
df["Churn"] = np.where(df["Churn"] == "Yes", 1, 0)

def target_summary_with_cat(dataframe, target, cat_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(cat_col)[target].mean()}), end="\n\n\n")

target_summary_with_cat(df,"Churn",cat_cols)

def target_summary_with_num(dataframe, target, num_col):
    print(pd.DataFrame({num_col: dataframe.groupby(target)[num_col].mean(),
                        "Count": dataframe.groupby(target)[num_col].count() }), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"Churn",col)

#5. Adım: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, variable, low_q=0.10, up_q=0.90):
    q1 = dataframe[variable].quantile(low_q)
    q3 = dataframe[variable].quantile(up_q)
    interquantile_range = q3 - q1
    up_lim = q3 + 1.5 * interquantile_range
    low_lim = q1 - 1.5 * interquantile_range
    return low_lim, up_lim

def replace_with_thresholds(dataframe, variable):
    low_lim, up_lim = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_lim), variable] = low_lim
    dataframe.loc[(dataframe[variable] > up_lim), variable] = up_lim
def check_outlier(dataframe, col_name):
    low_lim, up_lim = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_lim) | (dataframe[col_name] < low_lim)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(col, check_outlier(df, col))

# 6. Adım: Eksik gözlem analizi yapınız.
print(dataframe.isnull().sum())

# 7. Adım: Korelasyon analizi yapınız.

dataframe.corr(method="pearson")

#GÖREV 2:
# 1. Adım:  Eksik ve aykırı değerler için gerekli işlemleri yapınız.

include_zero = [print(col) for col in num_cols if 0 in df[col].values]
df.fillna(0,inplace=True)

# 2. Adım:  Yeni değişkenler oluşturunuz.
dataframe.tenure.describe()
dataframe.loc[(dataframe["tenure"] < 10), 'NEW_TENURE'] = 'New Customer'
dataframe.loc[(dataframe["tenure"] >= 10) & (dataframe["tenure"] < 25), 'NEW_TENURE'] = 'Potentials'
dataframe.loc[(dataframe["tenure"] >= 25) & (dataframe["tenure"] < 50), 'NEW_TENURE'] = 'Loyals'
dataframe.loc[(dataframe["tenure"] >= 50), 'NEW_TENURE'] = 'Champs'
dataframe.groupby('NEW_TENURE')["Churn"].mean()

dataframe["New_Total_Income"]=dataframe["tenure"]*dataframe["MonthlyCharges"]
dataframe.MonthlyCharges.describe()
dataframe.loc[(dataframe["MonthlyCharges"] < 40), 'NEW_MONTHLY_INCOME_CAT'] = 'Low'
dataframe.loc[(dataframe["MonthlyCharges"] >= 40) & (dataframe["MonthlyCharges"] < 70), 'NEW_MONTHLY_INCOME_CAT'] = 'Medium'
dataframe.loc[(dataframe["MonthlyCharges"] >= 70) & (dataframe["MonthlyCharges"] < 90), 'NEW_MONTHLY_INCOME_CAT'] = 'Ideal'
dataframe.loc[(dataframe["MonthlyCharges"] >= 90), 'NEW_MONTHLY_INCOME_CAT'] = 'High'
dataframe.groupby("NEW_MONTHLY_INCOME_CAT")["Churn"].mean()

# 3. Adım: Encoding işlemlerini gerçekleştiriniz.
cat_var,num_var,car_var=util.sep(data,10,20)
cat_var=[i for i in cat_var if i !="Outcome"]
data=util.one_hot_encoder(data,cat_deg)

#4. Adım: Numerik Değişkenler için Standarlaştırma Yapınız.
scaler = StandardScaler()
dataframe[num_cols] = scaler.fit_transform(dataframe[num_cols])

#5. Adım: Model Oluşturunuz.
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, random_state=42)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

best_models = models.sort_values(["F1 Score", "Accuracy"], ascending=False)[:3].index

print(best_models)
