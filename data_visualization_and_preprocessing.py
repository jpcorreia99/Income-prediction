import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
header = ["age","workclass","fnlwgt","education","education_num","marital_status","occupation","relationship","race","sex",
          "capital_gain","capital_loss","hours_per_week","native_country","income_category"]



df_train = pd.read_csv("unprocessed_data/adult.data", names = header, index_col = False)
df_train.drop("education",axis=1, inplace=True) #education-num is basically the same
df_test = pd.read_csv("unprocessed_data/adult.test", names = header, index_col = False, skiprows=1)
df_test.drop("education",axis =1,  inplace=True)

#two different ways of dealing with missing data
df_train.replace(' ?', np.nan, inplace=True)
df_test[df_test == ' ?'] = np.nan
df_test.replace(' >50K.', ' >50K', inplace = True)
df_test.replace(' <=50K.', ' <=50K', inplace = True)

#after analising we see that the missing values are in categorical collumns, so there's no use in filling nan's
print("Train data before droppin nan's:\n")
print(df_train.info())
print("\n\nTrain data before droppin nan's:\n")
print(df_train.info())

df_train.dropna(inplace = True)
df_test.dropna(inplace = True)
print("\n\nTrain data after droppin nan's:\n")
print(df_train.info()) #30162 elements
print("\n\nTest data after droppin nan's:\n")
print(df_test.info()) #15060

df_train_2 = df_train.copy() #this will be used later to
df_test_2 = df_test.copy()

#see correlation between numeric features
features_to_encode = ["workclass","marital_status","occupation","relationship","race","sex","native_country","income_category"]
#labels to encode
#Versão labelEncoded
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for feature in features_to_encode:
    encoded_feature= label_encoder.fit_transform(df_train[[feature]])
    df_train[feature] = encoded_feature
    encoded_feature = label_encoder.transform(df_test[feature])
    df_test[feature] = encoded_feature

#testing models

x_train = df_train.drop("income_category",axis=1) #inplace = True, altera diretamente os dados, False, retorna uma cópia
y_train =df_train[["income_category"]]
x_test = df_test.drop("income_category",axis=1)
y_test = df_test[["income_category"]]



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for feature in x_train.columns:
    x_train[feature] = scaler.fit_transform(x_train[[feature]])
    x_test[feature] = scaler.transform(x_test[[feature]])

from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(x_train, y_train)

label_encoded_predictions = random_forest_classifier.predict(x_test)
print("--- Metric when categorical features are just label encoded---")
print("accuracy: ", accuracy_score(y_test, label_encoded_predictions))
print("recall: ", recall_score(y_test, label_encoded_predictions))
print("f1 score: ", f1_score(y_test, label_encoded_predictions))
print("Confusion Matrix: \n", confusion_matrix(y_test, label_encoded_predictions))



# Versão One Hot
from sklearn.preprocessing import OneHotEncoder
features_to_encode = ["workclass","marital_status","occupation","relationship","race","sex","native_country"]


df_train_2["income_category"] = label_encoder.fit_transform(df_train_2[["income_category"]])
df_test_2["income_category"] = label_encoder.transform(df_test_2[["income_category"]])

# values will give the values in an array. (shape: (n,1)
# ravel will convert that array shape to (n, )
y_train_2 = df_train_2["income_category"].values.ravel()
x_train_2 = df_train_2.drop("income_category", axis =1)
y_test_2 = df_test_2["income_category"].values.ravel()
x_test_2 = df_test_2.drop("income_category", axis =1)

one_hot_encoder = OneHotEncoder()
numeric_features = ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week']
df_train_2 = x_train_2[numeric_features]
df_test_2 = x_test_2[numeric_features]

for feature in features_to_encode:
    df_train_2 = np.c_[df_train_2,one_hot_encoder.fit_transform(x_train_2[[feature]]).todense()]
    df_test_2 = np.c_[df_test_2, one_hot_encoder.transform(x_test_2[[feature]]).todense()]

random_forest_classifier.fit(df_train_2,y_train_2)
one_hot_predictions = random_forest_classifier.predict(df_test_2)

print("--- Metric when categorical features are one-hot-encoded---")
print("accuracy: ", accuracy_score(y_test, one_hot_predictions))
print("recall: ", recall_score(y_test, one_hot_predictions))
print("f1 score: ", f1_score(y_test, one_hot_predictions))
print("Confusion Matrix: \n", confusion_matrix(y_test, one_hot_predictions))



#Now that
corr_matrix = df_train.corr()
print("\nCorrelations between income_category:")
print(corr_matrix["income_category"].sort_values(ascending=False))

#we can drop fnlwgt

x_train = x_train.drop("fnlwgt", axis = 1)
x_test = x_test.drop("fnlwgt", axis =  1)

#save them all
x_train.to_pickle("preprocessed_data/x_train")
y_train.to_pickle("preprocessed_data/y_train")
x_test.to_pickle("preprocessed_data/x_test")
y_test.to_pickle("preprocessed_data/y_test")


