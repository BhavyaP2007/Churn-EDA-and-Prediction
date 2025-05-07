import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
df1 = pd.read_csv("test.csv")
df2 = pd.read_csv("train.csv")
df = pd.concat([df1,df2],axis=0)
pd.set_option("display.max_columns",None)
cols_one_hot = ["Gender","Subscription Type","Contract Length"]
x = df.iloc[:,:-1]
x.dropna(inplace=True)
y = df.iloc[:,-1:]
y.dropna(inplace=True)
# print(df.head(1))
encoder = ColumnTransformer([
    ("onehot",OneHotEncoder(sparse_output=False),cols_one_hot)
],remainder="passthrough")
x = encoder.fit_transform(x)
x = pd.DataFrame(x,columns=encoder.get_feature_names_out())
y = y.values.ravel()
# print(x.shape)
# print(y.shape)
# print(x.isna().sum())
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
params = {
    "n_estimators":list(range(100,300,5)),
    "max_depth":[None,10,20,30,40],
    "min_samples_leaf":[1,2,3,4],
    "min_samples_splot":[1,2,3,4,5,6,7,8],
    "bootstrap":[True,False]

}
# xgb = XGBClassifier()
# xgb.fit(X_train,y_train)
print(accuracy_score(y_train,rfc.predict(X_train)))
print(accuracy_score(y_test,rfc.predict(X_test)))
# print(rfc.get_params())
