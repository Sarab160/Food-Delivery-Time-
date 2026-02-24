import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df=pd.read_csv("Food.csv")

print(df.head())
print(df.info())

df["Weather"]=df["Weather"].fillna(df["Weather"].mode())
df["Traffic_Level"]=df["Traffic_Level"].fillna(df["Traffic_Level"].mode())
df["Time_of_Day"]=df["Time_of_Day"].fillna(df["Time_of_Day"].mode())
df["Courier_Experience_yrs"]=df["Courier_Experience_yrs"].fillna(df["Courier_Experience_yrs"].mode())

print(df.info())

# sns.pairplot(df,hue="Delivery_Time_min")
# plt.show()

print(df.columns)

x=df[["Distance_km","Preparation_Time_min"]]
y=df["Delivery_Time_min"]

features=df[["Weather","Traffic_Level","Time_of_Day","Vehicle_Type","Courier_Experience_yrs"]]
ohe=OneHotEncoder(sparse_output=False,drop="first")
encode_data=ohe.fit_transform(features)
encode_dataframe=pd.DataFrame(encode_data,columns=ohe.get_feature_names_out(features.columns))

x_final=pd.concat([x,encode_dataframe],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x_final,y,test_size=0.2,random_state=42)

knc=LinearRegression()
knc.fit(x_train,y_train)

print("Test Score: ",knc.score(x_test,y_test))

y_pred = knc.predict(x_test)

print("Mean absolute error: ",mean_absolute_error(y_test,y_pred))
