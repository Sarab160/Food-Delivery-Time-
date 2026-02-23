import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split

df=pd.read_csv("Food.csv")

print(df.head())
print(df.info())

df["Weather"]=df["Weather"].fillna(df["Weather"].mode)
df["Traffic_Level"]=df["Traffic_Level"].fillna(df["Traffic_Level"].mode)
df["Time_of_Day"]=df["Time_of_Day"].fillna(df["Time_of_Day"].mode)
df["Courier_Experience_yrs"]=df["Courier_Experience_yrs"].fillna(df["Courier_Experience_yrs"].mode)

print(df.info())

# sns.pairplot(df,hue="Delivery_Time_min")
# plt.show()

print(df.columns)

x=df[["Distance_km","Preparation_Time_min"]]
y=df["Delivery_Time_min"]


