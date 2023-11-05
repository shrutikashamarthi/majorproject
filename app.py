import streamlit as st
import joblib
import json 
import os
from collections import Counter
import base64


rf = joblib.load(os.path.join('files','rf.sav'))
knn = joblib.load(os.path.join('files','knn.sav'))
lr = joblib.load(os.path.join('files','lr.sav'))
dt = joblib.load(os.path.join('files','dt.sav'))
mnb = joblib.load(os.path.join('files','nb.sav'))
vect = joblib.load(os.path.join('files','vect.sav'))
with open(os.path.join('files','stopwords.json'),'r') as fp:
    stopwords = json.load(fp)


def preprocess_descp(descp):
    descp = ' '.join([word for word in descp.split(' ') if word not in list(stopwords.values())])
    arr = vect.transform([descp])
    return arr  
    
def model_prediction(model,arr):
    prob = model.predict_proba(arr).tolist()[0]
    if prob[0] > prob[1]:
        prob = prob[0]
    else:
        prob = prob[1]
    pred = model.predict(arr)[0]  
    return round(prob,2),pred

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
set_bg_hack('bg.jpg')

# st.title("Real vs Fake Job Predictor App")
st.markdown("<h1 style='color: black; font-size: 36px;'>Real vs Fake Job Predictor App</h1>", unsafe_allow_html=True)
user = st.text_input("Enter Name")
job_description = st.text_area("Enter Job Description")

if st.button("Predict"):
    if user and job_description:
        arr = preprocess_descp(job_description)
        
        rf_prob,rf_pred = model_prediction(rf,arr)
        if rf_pred == 0:
            st.success(f"Random Forest Prediction (best model): {rf_pred,rf_prob}")
        else:
            st.error(f"Random Forest Prediction (best model): {rf_pred,rf_prob}")
        
        
        mnb_prob,mnb_pred = model_prediction(mnb,arr)
        if mnb_pred == 0:
            st.success(f"Naive Bayes Prediction: {mnb_pred,mnb_prob}")
        else:
            st.error(f"Naive Bayes Prediction: {mnb_pred,mnb_prob}")
            
        dt_prob,dt_pred = model_prediction(dt,arr)
        if dt_pred == 0:
            st.success(f"Decision Tree Prediction: {dt_pred,dt_prob}")
        else:
            st.error(f"Decision Tree Prediction: {dt_pred,dt_prob}")
        
        
        knn_prob,knn_pred = model_prediction(knn,arr)
        if knn_pred == 0:
            st.success(f"KNN Prediction: {knn_pred,knn_prob}")
        else:
            st.error(f"KNN Prediction: {knn_pred,knn_prob}")
        
        lr_prob,lr_pred = model_prediction(lr,arr)
        if lr_pred == 0:
            st.success(f"Logistic Regression Prediction: {lr_pred,lr_prob}")
        else:
            st.error(f"Logistic Regression Prediction: {lr_pred,lr_prob}")
        
        
        
        final_pred = Counter([rf_pred,mnb_pred,knn_pred,lr_pred,dt_pred]).most_common(1)[0][0]
        # final_prob = (rf_prob+mnb_prob+dt_prob+knn_prob+lr_prob)/5
        if final_pred == 0:
            st.success(f"Cumulative Prediction: {final_pred}")
        else:
            st.error(f"Cumulative Prediction: {final_pred}")

    else:
        st.warning("Please enter both Username and Job Description.")

