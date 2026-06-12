import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('ford.csv')
#EDA(EXPLORATORY DATA ANALYSIS)
#print(df.shape)
#print(df.head())
#print(df.describe())
#print(df.info())
#print(df.isnull().sum())
#sns.histplot(df['price'],bins=50,kde=True)
#plt.show()
#sns.heatmap(df.corr(numeric_only=True),annot=True)
#plt.show()
#sns.boxplot(data=df,x='year',y='price')
#plt.show()
#sns.scatterplot(data=df,x='mileage',y='price')
#plt.show()
#sns.boxplot(data=df,x='engineSize',y='price')
#plt.show()
#sns.boxplot(data=df,x='transmission',y='price')
#plt.show()
#sns.boxplot(data=df,x='fuelType',y='price')
#plt.show()
x=df.drop(columns=['price'])
y=df['price']
x_encoded=pd.get_dummies(x,columns=['model','transmission','fuelType'])
x_encoded=x_encoded.astype(int)
from sklearn.preprocessing import StandardScaler
cols=['year','mileage','tax','mpg','engineSize']
scaler=StandardScaler()
x_encoded[cols]=scaler.fit_transform(x_encoded[cols])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(   x_encoded, y, test_size=0.20, random_state=42)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)#to test accuracy of our model
n=x_test.shape[0]
p=x_test.shape[1]
r2_adjusted=1-((1-r2)*(n-1)/(n-p-1))
print(r2_adjusted)
print(r2)

#Now we can deploy our model as our accuracy is great
import joblib

joblib.dump(model, "ford_model.pkl")
joblib.dump(scaler, "ford_scaler.pkl")

# Save feature order
joblib.dump(x_encoded.columns.tolist(), "feature_columns.pkl")

print("Model saved successfully!")
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(mae)
print(df['model'].unique())
print(df['transmission'].unique())
print(df['fuelType'].unique())






