import streamlit as st
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
#import re
from PIL import Image

with open('model1.pkl', "rb") as f:
	model1 = pickle.load(f)
with open('model2.pkl', "rb") as f:
	model2 = pickle.load(f)
with open('model3.pkl', "rb") as f:
	model3 = pickle.load(f)
with open('model4.pkl', "rb") as f:
	model4 = pickle.load(f)
with open('model5.pkl', "rb") as f:
	model5 = pickle.load(f)
	
!pip install xgboost

@st.cache()

# def load_models(path):
# 	"""
# 	Load models from pickle files for each of the 5 columns to be predicted
# 	parameters: 
# 		path: location where pickle model files are stored
# 	return:
# 		model1, model2, model3, model4, model5: trained model
# 	"""
# 	with open(f'{path}/model1.pkl', "rb") as f:
# 		model1 = pickle.load(f)
# 	with open(f'{path}/model2.pkl', "rb") as f:
# 		model2 = pickle.load(f)
# 	with open(f'{path}/model3.pkl', "rb") as f:
# 		model3 = pickle.load(f)
# 	with open(f'{path}/model4.pkl', "rb") as f:
# 		model4 = pickle.load(f)
# 	with open(f'{path}/model5.pkl', "rb") as f:
# 		model5 = pickle.load(f)

# 	return model1, model2, model3, model4, model5


def validate_input(user_input):
	"""
	Custom function to validate DNA sequence input
	parameters:
		user_input: Spacer and PAM sequence entered in text input
	"""

	if len(user_input) == 0:
		st.session_state.error = "No input entered."

	elif len(user_input) != 23:
			st.session_state.error  = "Length of DNA sequence should be 23 characters."

	else:
		if re.search(r'[^ACGT]', user_input):
			st.session_state.error = "Only characters of A, C, G or T should be entered"


def main():
	def one_hot_encode_sequence(S):
		encoding = np.zeros((len(S), num_chars), dtype=int)
		for i, char in enumerate(S):
			encoding[i, char_to_int[char.upper()]] = 1
		return encoding


	def prediction(X):
		"""
		Get predicted value for each column by passing user input to the trained models
		parameters:
			X: Spacer and PAM sequence from text input
		return:
			df_result: Dataframe of one row containing values predicted by the models
		"""
		predictions = []
		predictions.append(f'{round(float(model1.predict(X)[0])*100,2)}%')
		predictions.append(f'{round(float(model2.predict(X)[0]),3)} bps')
		predictions.append(f'{round(float(model3.predict(X)[0]),3)} bps')
		predictions.append(f'{round(float(model4.predict(X)[0]),3)} bps')
		predictions.append(f'{round(float(model5.predict(X)[0])*100,2)}%')

		df_result = pd.DataFrame({'Fraction of total reads with insertion': [predictions[0]], 'Average Insertion Length': [predictions[1]], 
				'Average Deletion Length': [predictions[2]], 'Indel Diversity': [predictions[3]], 'Fraction Frameshifts': [predictions[4]]})
		
		return df_result


	# Page config
	apptitle = 'CRISPR Repair Outcome Prediction'
	st.set_page_config(page_title=apptitle, page_icon='random', layout= 'wide', initial_sidebar_state="expanded")
	st.title('CRISPR Repair Outcome Prediction Tool')

	#path = r'https://github.com/tamshan/Streamlit.git'
	#model1, model2, model3, model4, model5 = load_models(path)

	# page HTML formatting
	image = Image.open('CRISPR.jpg')
	new_image = image.resize((1200, 200))
	st.image(new_image, use_column_width=True)
	
	css = '''
	<style>
	section.main > div:has(~ footer ) {
		padding-bottom: 5px;
	}
	</style>
	'''
	st.markdown(css, unsafe_allow_html=True)
    
    # DNA Sequence Input
	user_input = st.text_input(
		label="Enter a Spacer followed by a PAM Sequence (23 characters of A,C,G,T only):", 
		max_chars=23,
		placeholder="CACGCTGTCATCCACCAGGTAGG",
		key = 'input'
	)
    
	# Create a mapping of DNA characters to integers
	dna_chars = 'ACGT'
	char_to_int = dict((c, i) for i, c in enumerate(dna_chars))
	num_chars = len(dna_chars)

	# prediction  button
	if st.button("Predict", on_click=validate_input, args=(user_input,)):
		st.write("Your input:", user_input)
		try:
			X = np.array(one_hot_encode_sequence(user_input))
			X = X.reshape(1,92)
			assessment = prediction(X)
			st.dataframe(assessment, hide_index=True)
		except:
			st.error(st.session_state.error)

	st.markdown(""" <style>
	#MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
	</style> """, unsafe_allow_html=True)

	padding = 0
	st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


if __name__ == '__main__':
	main()
