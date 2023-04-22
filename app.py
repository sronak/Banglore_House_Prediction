import pandas as pd
from flask import Flask, render_template,request
import pickle 
import numpy as np

# import cgi
# form = cgi.FieldStorage()


app = Flask(__name__,)
data = pd.read_csv('Clean_Dataset.csv')
pipe = pickle.load(open('xgboost_model.pkl','rb'))

@app.route('/')
def HP_Form():
	locations = data['location'].unique()
	area_type = data['area_type'].unique()
	size = data['bhk'].unique()
	bath = data['bath'].unique()
	balcony =  data['balcony'].unique()
	return render_template('HP_Form.html',locations = locations,area_types = area_type,bath = bath,size = size)
	# write loop in html page to fill spots
    
@app.route('/predict',methods =['POST'])
def predict():

	# print(locations)
	
	bhk = float(request.form.get('size'))
	area_type = request.form.get('area')
	total_sqft = float(request.form.get('vol'))	
	bath = float(request.form.get('bath'))
	locations = request.form.get('location') #element id/name from wen elements
	list1 = [locations,total_sqft,bath,bhk]

	# locations =  form.getvalue('location')
	# print(locations)
	# print(locations,bhk,balcony,total_sqft,total_sqft) # print passed data
    
    # match the columns of both file
	input = pd.DataFrame([[locations,total_sqft,bath,bhk]],columns = ['location','total_sqft','bath','bhk'])
	prediction = pipe.predict(input)[0]

	return str(np.round(prediction,2))
	# return list1


if __name__ == "__main__":
	app.run(debug = True,port = 5000)
