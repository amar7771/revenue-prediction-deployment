import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
model=pkl.load(open(r'C:\Users\atefr\Downloads\linear_regression_model.pkl','rb'))
features=['City', 'Number of Referrals', 'Tenure in Months', 'Phone Service','Offer',
       'Avg Monthly Long Distance Charges', 'Multiple Lines',
       'Internet Service', 'Internet Type', 'Avg Monthly GB Download',
       'Online Security', 'Online Backup', 'Device Protection Plan',
       'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
       'Streaming Music', 'Unlimited Data', 'Paperless Billing',
       'Payment Method', 'Total Extra Data Charges',
       'Total Long Distance Charges']


cat_features = [
    'Gender', 'Married', 'Phone Service', 'Multiple Lines', 'Internet Service','Offer',
    'Online Security', 'Online Backup', 'Device Protection Plan',
    'Streaming Music', 'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
    'Unlimited Data', 'Paperless Billing', 'Internet Type', 'Contract', 'Payment Method'
]
target='City'

def load_encoder(feature):
    return pkl.load(open(f'C:\\Users\\atefr\\Downloads\\{feature}_encoder.pkl','rb'))

target_encoder=pkl.load(open(f'C:\\Users\\atefr\\Downloads\\city_target_mean.pkl','rb'))


def main():
    st.set_page_config(layout='wide')
    st.title('Revnue predction features')
    inputs={}
    for f in features:
        if f==target:
                options = list(target_encoder.index)
                inputs[f] = st.selectbox(f, options)
        if f in cat_features:
            en=load_encoder(f)
            if hasattr(en, 'categories_'):
                    options = en.categories_[0].tolist()
                    inputs[f] = st.selectbox(f, options)
            elif hasattr(en, 'classes_'):
                    options = en.classes_.tolist()
                    inputs[f] = st.selectbox(f, options)
            
                 
        else:
             if f!=target:     
                inputs[f]=st.number_input(f)
    if st.button('predict'):
        input_values = []
        for f in features:
            value = inputs[f]
            if f in cat_features:
                en = load_encoder(f)
                value = en.transform(np.array([[value]]))[0]
            elif f == target:
                value = target_encoder[value]  # map city name to encoded value
            input_values.append(value)
        
        input_values = np.array(input_values, dtype='object').reshape(1, -1)
        y_predict = model.predict(input_values)
        st.success(f"Predicted Revenue: {y_predict[0]:,.2f}")
if __name__=='__main__':
     main()
            
             
                      