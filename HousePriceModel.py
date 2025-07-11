import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import pickle


file=pd.read_csv("C:\\Users\\USER\\Downloads\\4th sem\\py\\Housing.csv")

# print(file.head(5))

# print(file.isnull().sum())

# print(file.describe())

print(file.columns)

print(file.dtypes)

mainroad_le=LabelEncoder()
guestroom_le=LabelEncoder()
basement_le=LabelEncoder()
hotwaterheating_le=LabelEncoder()
airconditioning_le=LabelEncoder()
furnishingstatus_le=LabelEncoder()
prefarea_le=LabelEncoder()

file['mainroad']=mainroad_le.fit_transform(file['mainroad'])
file['guestroom']=guestroom_le.fit_transform(file['guestroom'])
file['basement']=basement_le.fit_transform(file['basement'])
file['hotwaterheating']=hotwaterheating_le.fit_transform(file['hotwaterheating'])
file['airconditioning']=airconditioning_le.fit_transform(file['airconditioning'])
file['prefarea']=prefarea_le.fit_transform(file['prefarea'])
file['furnishingstatus']=furnishingstatus_le.fit_transform(file['furnishingstatus'])


# sns.heatmap(file.corr(numeric_only=True),annot=True)

# plt.show()

y=file['price']

file.drop(columns=['price','furnishingstatus','prefarea','hotwaterheating'],axis=1,inplace=True)

x=file

# x=df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'airconditioning', 'parking']]
# y=df['price']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestRegressor()

model.fit(x_train,y_train)

y_pre=model.predict(x_test)

print("r2_score : ",r2_score(y_test,y_pre))
print("mean_squared_error:",mean_squared_error(y_test,y_pre))

with open("HousePriceModel.pkl","wb" ) as f:
    pickle.dump(model,f)

with open("mainroad_le.pkl","wb") as f:
    pickle.dump(mainroad_le,f)

with open("guestroom_le.pkl","wb") as f:
    pickle.dump(guestroom_le,f)

with open("basement_le.pkl","wb") as f:
    pickle.dump(basement_le,f)

with open("airconditioning_le.pkl","wb") as f:
    pickle.dump(airconditioning_le,f)






















