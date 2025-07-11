import pandas as pd
import numpy as np
from flask import Flask,request,jsonify
from flask_cors import CORS
import pickle

app=Flask(__name__)

CORS(app)

with open("HousePriceModel.pkl","rb") as f:
    model=pickle.load(f)

with open("basement_le.pkl","rb") as f:
    basement_le=pickle.load(f)

with open("guestroom_le.pkl","rb") as f:
    guestroom_le=pickle.load(f)

with open("mainroad_le.pkl","rb") as f:
    mainroad_le=pickle.load(f)

with open("airconditioning_le.pkl","rb") as f:
    airconditioning_le=pickle.load(f)






@app.route("/")
def home():
    return "welcome to Home page"


@app.route("/predict",methods=["POST"])
def predict():
    try:
        data=request.get_json()

        if not data:
            return jsonify({"error":"data is not found"})
    
        df=pd.DataFrame([data])
        
        #encoding the cat property

        # df['mainroad']=mainroad_le.transform(df['mainroad']) 
        # df['mainroad'] is a single string, not a list or array.
        # LabelEncoder.transform() expects a 1D list/array-like, not a string.

        # df['mainroad'][0]     # gives 'yes'
        # [df['mainroad'][0] ] # becomes ['yes']

        #  note:check always frontend in which form data is sending if data is convert into numer into frontend then no use of label encoding and it will give error because label encoding train on string but now it getting number  
      
        df['mainroad']=mainroad_le.transform([df['mainroad'][0]]) 
        df['basement']=basement_le.transform([df['basement'][0]])
        df['guestroom']=guestroom_le.transform([df['guestroom'][0]])
        df['airconditioning']=airconditioning_le.transform([df["airconditioning"][0]])

        features=df[['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'airconditioning', 'parking']]

        pred_val=model.predict(features)
        # it return data in array

        return jsonify({"price":round(pred_val[0],2)})
    
    except Exception as e:
         return jsonify({"error": str(e)})

    

if __name__=="__main__":
    app.run(debug=True)




