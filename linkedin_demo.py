import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.markdown('## Enter demographic data to predict if a person is a LinkedIn user')

model = joblib.load('linkedin_model.pkl')

age = st.slider('Select your age:', 1, 100)

edu_lookup = {'<HS':1, 'HS incomplete':2, 'HS complete':3, 'College incomplete':4, 'Associates':5, 'College complete':6, 'Grad incomplete':7, 'Grad complete':8, "Unknown":98, 'Refuse':99}
edu = st.radio('Education level:', ['<HS', 'HS incomplete', 'HS complete', 'College incomplete', 'Associates', 'College complete', 'Grad incomplete', 'Grad complete', "Unknown", 'Refuse'])
edu = edu_lookup[edu]

income_lookup = {'<10k':1, '10-20k':2, '20-30k':3, '30-40k':4, '40-50k':5, '50-75k':6, '75-100k':7, '100-150k':8, '150k+':9, "Unknown":98, 'Refuse':99}
income = st.radio('Income level:', ['<10k', '10-20k', '20-30k', '30-40k', '40-50k', '50-75k', '75-100k', '100-150k', '150k+', "Unknown", 'Refuse'])
income = income_lookup[income]

marital = st.radio('Married:', ['Yes', 'No'])
parent = st.radio('Have children:', ['Yes', 'No'])
gender = st.radio('Gender:', ['Female', 'Male'])


if marital == 'Yes':
	marital = 1
else:
	marital = 0

if parent == 'Yes':
	parent = 1
else:
	parent = 0

if gender == 'Female':
	gender = 1
else:
	gender  = 0

st.markdown(f'age {age}')
st.markdown(f'edu {edu}')
st.markdown(f'income {income}')
st.markdown(f'marital {marital}')
st.markdown(f'parent {parent}')
st.markdown(f'gender {gender}')

if st.button('Predict'):
	pre_pred = pd.DataFrame({ 'age':[age],   'marital':[marital],   'par':[parent],   'educ2':[edu],   'income':[income],   'female':[gender]  })
	user_prob = model.predict_proba(pre_pred)
	user_prob = round( user_prob[0], 2 ) * 100
	
	st.markdown(f'The probability that the entered person uses LinkedIn is {user_prob}%')
	#st.markdown(f'### Prediction is {pred}')
		




