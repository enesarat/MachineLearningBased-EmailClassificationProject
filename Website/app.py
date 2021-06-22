from flask import Flask, render_template, request, redirect, url_for
import os
from os.path import join, dirname, realpath
import pandas as pd
import csv
import pickle
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

app.config["DEBUG"] = True

data = None

@app.route("/", methods=['GET','POST'])
def uploadFiles():
		check = -1
		
		# multinominal
		model1 = pickle.load(open(r'mnb.pkl','rb')) # path of multinominal model
		#svc
		model2 = pickle.load(open(r'svc.pkl','rb')) # path of svc model
		# logistic regression
		model3 = pickle.load(open(r'lr.pkl','rb')) # path of logistic regression model
		
		cv = pickle.load(open(r'vectorizer.pkl','rb')) # path of count vectorizer model
	
		msg = str(request.form.get('email'))
		data=[msg]
		vect = cv.transform(data).toarray()

		new_prediction = model1.predict(vect)
		new_prediction2 = model2.predict(vect)
		new_prediction3 = model3.predict(vect)
		
		total_acc_h=0
		total_acc_s=0
		predict_result=0
		predict_value=0

		if new_prediction[0]==1:
			total_acc_h+= 1
		else:
			total_acc_s+= 1
		if new_prediction2[0]==1:
			total_acc_h+= 1
		else:
			total_acc_s+= 1
		if	new_prediction3[0]==1:
			total_acc_h+=1
		else:
			total_acc_s+= 1

		if (total_acc_h/3)>=(total_acc_s/3):
			predict_result=1
			predict_value=(total_acc_h/3)
		else:
			predict_result=0
			predict_value=(total_acc_s/3)

		
		if request.method == 'POST':
			check=1;		
		
		predict_value=float("{:.2f}".format((predict_value * 100) - 1))

		#return render_template('data.html',predict_result=predict_result,predict_value=predict_value,check=check)

		return render_template('data.html',predict_result=predict_result,check=check)


if (__name__ == "__main__"):
     app.run(port = 5000)


