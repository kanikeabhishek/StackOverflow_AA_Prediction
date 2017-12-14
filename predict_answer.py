import pandas as pd
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
from textblob import TextBlob
import xml.etree.ElementTree as ET
from textstat.textstat import textstat
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier



SO_DATASET = os.path.join(os.path.dirname(__file__), 'data', 'stackoverflow_dataset.csv')
SO_ANSWERS = os.path.join(os.path.dirname(__file__), 'answer.csv')

"""
load stackoverflow dataset into a DataFrame
input : file path of the dataset
return : pandas dataframe
"""
def load_so_corpus(dataset):
	return pd.read_csv(dataset)
"""
load preprocessed data from answers.csv
return : pandas DataFrame
"""
def load_so_answers():
	return pd.read_csv(SO_ANSWERS, index_col=0)
"""
removes html tags such as <code>,<a>...
input : string consisting of question_body or answer_body
return : string with all html tags removed

"""
def removehtmltags(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

"""
Preprocess data inorder to calculate readability of the answer and similarity between question and answer

"""
def do_data_preprocessing():

	col = ['READABILITY','ANSWER_SCORE','ANSWERER_SCORE','SIMILARITY','TIME_DIFF','POLARITY','LABEL']
	final_feature_df = pd.DataFrame(columns = col)
	df = load_so_corpus(SO_DATASET)
	# If answer.csv is present with required number of rows return
	#if load_so_answers().shape[0] == pickle.load(open("saver_pickle", "rb")).shape[0] - 1:
		#return

	################################## Save stackoverflow corpus as pickle ##################################
	'''
	df = load_so_corpus(SO_DATASET)
	pickle.dump(df, open("saver_pickle", "wb"))
	'''
	################################## Save stackoverflow corpus as pickle ##################################

	# Load stackoverflow corpus from pickle
	#df = pickle.load(open("saver_pickle", "rb"))

	# Answer dataframe
	#answer_df = pd.DataFrame(columns = ["label", "readability", "answer_score", "User_reputation", "similarity"])

	# Readibility accuracy measure variables true: accepted variable, false: other answers varaible
	numerator_true = 0
	denominator_true = 0
	numerator_false = 0
	denominator_false = 0

	# Similarity accuracy measure variables
	sim_num = 0
	sim_deno = 0
	tokenize = lambda doc: doc.lower().split(" ")
	tfidf_vector = TfidfVectorizer(norm = 'l2',min_df = 0,stop_words = "english",use_idf = True,ngram_range=(1,3),sublinear_tf=True,tokenizer = tokenize)

	# Main loop which runs overall rows of corpus
	for answer in range(df.shape[0]):
		# Question ID column is saved as filename, since pandas is reading from .tar.gz (zipped) format
		'''if np.isnan(df["newsample-000000000000.csv"][answer]):
			continue
		question_id = int(df["newsample-000000000000.csv"][answer])'''
		comment_text = df['COMMENT'][answer]
		# print ("comment: {} type: {}".format(comment_text, type(comment_text)))
		if(type(comment_text) != float):
			blob = TextBlob(comment_text)
			polarity = blob.polarity
		else:
			continue

		# Label feature addition
		label = (1 if df["A_ID"][answer] == df["Accepted_Answer_ID"][answer] else 0)

		################################ Measure Readibility of answers ################################
		answer_body = df["A_BODY"][answer]
		soup = BeautifulSoup(answer_body, "lxml")

		try:
			# Assumption1: Replace any <code> text by a 100% readable code - "A good working code" (Readability measure - 100%)
			if soup.find('code'):
				for _ in range(len(soup.find_all('code'))):
					if soup.find('code'):
						soup.find('code').replaceWith('A good working code')
					else:
						break

			# Assumption2: Replace any <a> text by a 100% readable code - "A working link" (Readability measure - 100%)
			if soup.find('a'):
				for _ in range(len(soup.find_all('a'))):
					if soup.find('a'):
						soup.find('a').replaceWith('A working link')
					else:
						break
		except Exception as e:
			# Check for any error
			print ("answer: " + df["A_BODY"][answer])
			raise e

		if soup != "":
			readability = textstat.flesch_reading_ease(soup.text)
		else:
			if label:
				readability = 100.0
			else:
				readability = 0.0
		################################ Measure Readibility of answers ################################

		################################ Measure Similarity of QnA ####################################

		question_body = df['Q_BODY'][answer]
		answer_body = df['A_BODY'][answer]
		#print(type(question_body),type(answer_body))
		#print(question_body,answer_body)
		q_text = removehtmltags(question_body)
		a_text = removehtmltags(answer_body)
		#print(q_text)
		QA_vector= tfidf_vector.fit_transform([q_text,a_text])
		#print(A.shape,Q.shape)
		Q_vector = QA_vector[0].toarray()
		A_vector = QA_vector[1].toarray()
		#print(Q_vector,A_vector)
		cs = cosine_similarity(Q_vector,A_vector)
		#print(cs)
		similarity = cs[0][0]
		#print(similarity)
		if (df["U_REPUTATION"][answer]):
			final_feature_df.loc[answer]  = [readability, df["A_SCORE"][answer], df["U_REPUTATION"][answer],similarity,df['TIME_DIFF'][answer],polarity,label]
		else:
			final_feature_df.loc[answer]  = [readability, df["A_SCORE"][answer],0,similarity,df['TIME_DIFF'][answer],polarity,label]
		################################ Measure Similarity of QnA ####################################


		###################### Accuracy calculations while measuring the features #####################
		if (label):
			denominator_true += 1
			sim_deno += 1
		if (readability > 20 and label == 1):
			numerator_true += 1

		if (similarity > 0.2 and label == 1):
			sim_num += 1

		if (label == 0):
			denominator_false += 1
		if (readability < 50 and label == 0):
			numerator_false += 1

		if (answer % 1000 == 0):
			print ("Answer Proccessed: " + str(answer))
			print ("Accepted answer... " + str(100*(numerator_true + 1)/float(denominator_true + 1)))
			print ("Other answer accuracy so far... " + str(100*(numerator_false + 1)/float((denominator_false + 1))))

			print ("Similarity accuracy so far... " + str(100*(sim_num + 1)/float((sim_deno + 1))))
			print ("-"*150)
		###################### Accuracy calculations while measuring the features #####################

	final_feature_df.to_csv("answer.csv")

"""
Classifies answers in test data as accepted answer or not . Various binary classifier are implemented using
sklearn library.

"""
def predictLabel():

	feature_list = ['Readability','Answer_score','Answerer_score','Similarity','Time_diff','Polarity']
	#fig,ax = plt.subplots()
	dtm = load_so_answers()
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(dtm.iloc[:, :-1].corr())
	ax.set_xticklabels(['']+feature_list)
	ax.set_yticklabels(['']+feature_list)

	fig.colorbar(cax)
	plt.show()

	accuracy_list = []

	#import seaborn as sns
	#corr = dtm.corr()
	#sns.heatmap(corr,
	#            xticklabels=corr.columns.values,
	#            yticklabels=corr.columns.values)
	# Seperate training set - Considering till last but 1000 answers
	train_data = dtm[:(dtm.shape[0] - 1000)]
	# Seperate test set - Considering last 1000 answers
	test_data = dtm[(dtm.shape[0] - 1000):]


	# ,READABILITY,ANSWER_SCORE,ANSWERER_SCORE,SIMILARITY,TIME_DIFF,POLARITY,LABEL
	# Train set: Drop label column from dataset
	actual_train_data = train_data.drop('LABEL', 1)
	actual_train_data.drop('ANSWERER_SCORE',axis = 1,inplace = True)
	actual_train_data.drop('ANSWER_SCORE',axis = 1,inplace = True)
	#actual_train_data.drop('POLARITY',axis = 1,inplace = True)
	# actual_train_data.drop('READABILITY',axis = 1,inplace = True)
	# actual_train_data.drop('SIMILARITY',axis = 1,inplace = True)
	#actual_train_data.drop('Id',axis = 1,inplace = True)
	# Train set: Label values
	actual_train_label = train_data['LABEL'].values
	#print(actual_train_data)

	# Test set: Drop label column from dataset
	actual_test_data = test_data.drop('LABEL', 1)
	actual_test_data.drop('ANSWERER_SCORE',axis = 1,inplace = True)
	actual_test_data.drop('ANSWER_SCORE',axis = 1,inplace = True)
	#actual_test_data.drop('POLARITY', axis= 1, inplace = True)
	# actual_test_data.drop('READABILITY', axis= 1, inplace = True)
	# actual_test_data.drop('SIMILARITY', axis= 1, inplace = True)
	actual_test_label = train_data['LABEL'].values
	# Test set: Label values
	actual_test_label = test_data['LABEL'].values



	# Baseline Classifier
	clf = DummyClassifier()
	clf.fit(actual_train_data, actual_train_label)
	predictions = clf.predict(actual_test_data)


	# Calculate accuracy
	numerator = 0
	for x in range(1000):
		if predictions[x] == actual_test_label[x]:
			numerator += 1
	print ("Prediction accuracy for baseline Classifier... " + str(100*numerator/1000.0))
	print(f1_score(actual_test_label,predictions,average= 'binary'))
	# Baseline Classifier
	accuracy_list.append(numerator/1000.0 * 100)


	# Ensemble Adaboost classifier
	clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),
	                         algorithm="SAMME",
	                         n_estimators=200)

	clf.fit(actual_train_data, actual_train_label)
	predictions = clf.predict(actual_test_data)
	numerator = 0
	for x in range(1000):
		if predictions[x] == actual_test_label[x]:
			numerator += 1
	print ("Prediction accuracy AdaBoostClassifier... " + str(100*numerator/1000.0))
	print(f1_score(actual_test_label,predictions,average= 'binary'))
	accuracy_list.append(numerator/1000.0 * 100)

	# Ensemble Adaboost classifier


	# Random Forest classifier
	clf = RandomForestClassifier(n_estimators=5)
	clf.fit(actual_train_data, actual_train_label)
	predictions = clf.predict(actual_test_data)
	numerator = 0
	for x in range(1000):
		if predictions[x] == actual_test_label[x]:
			numerator += 1
	print ("Prediction accuracy RandomForestClassifier... " + str(100*numerator/1000.0))
	print(f1_score(actual_test_label,predictions,average= 'binary'))
	accuracy_list.append(numerator/1000.0 * 100)

	# Random Forest classifier


	# Multilayer Perceptron classifier
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 1), random_state=1,activation = 'logistic')
	clf.fit(actual_train_data, actual_train_label)
	predictions = clf.predict(actual_test_data)
	numerator = 0
	for x in range(1000):
		if predictions[x] == actual_test_label[x]:
			numerator += 1
	print ("Prediction accuracy on test data NN... " + str(100*numerator/1000.0))
	print(f1_score(actual_test_label,predictions,average= 'binary'))
	accuracy_list.append(numerator/1000.0 * 100)



	# Bernoulli classifier

	clf = GaussianNB()
	#clf = Multi(alpha=1.0)

	# Fit the train data
	clf.fit(actual_train_data, actual_train_label)

	# Predict the test data
	predictions = clf.predict(actual_test_data)

	# Calculate accuracy
	numerator = 0
	for x in range(1000):
		if predictions[x] == actual_test_label[x]:
			numerator += 1
	print ("Prediction accuracy Naive Bayes... " + str(100*numerator/1000.0))
	print(f1_score(actual_test_label,predictions,average= 'binary'))
	accuracy_list.append(numerator/1000.0 * 100)


	clf = tree.DecisionTreeClassifier(max_depth = 5)
	clf.fit(actual_train_data, actual_train_label)
	predictions = clf.predict(actual_test_data)

	numerator = 0
	for x in range(1000):
		if predictions[x] == actual_test_label[x]:
			numerator += 1
	print ("Prediction accuracy Decision Trees... " + str(100*numerator/1000.0))
	print(f1_score(actual_test_label,predictions,average= 'binary'))
	accuracy_list.append(numerator/1000.0 * 100)

	fig,ax = plt.subplots()
	x = np.arange(6)
	plt.bar(x,accuracy_list,0.6)
	plt.xticks(x,('Baseline','Adaboost','Random Forest','Neural Network','Naive Bayes','Decision Tree'),fontsize = 12)
	ax.set_xlabel('Classifiers',fontsize = 14,weight= 'bold')
	ax.set_ylabel('Accuracy',fontsize = 14,weight = 'bold')
	plt.show()

def main():
	do_data_preprocessing()
	predictLabel()

if __name__ == '__main__':
	main()
