import pandas as pd
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
#from textblob import TextBlob
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


SO_DATASET = os.path.join(os.path.dirname(__file__), 'data', 'new_stackoverflow_dataset.csv')
SO_ANSWERS = os.path.join(os.path.dirname(__file__), 'answer.csv')

# Load Dataset
def load_so_corpus(dataset):
	return pd.read_csv(dataset)

def load_so_answers():
	return pd.read_csv(SO_ANSWERS, index_col=0)

def removehtmltags(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

def calculate_cosine_similarity():
    df = load_data()
    needed_data = df.iloc[:,4:6]
    tokenize = lambda doc: doc.lower().split(" ")
    tfidf_vector = TfidfVectorizer(norm = 'l2',min_df = 0,stop_words = "english",use_idf = True,ngram_range=(1,3),sublinear_tf=True,tokenizer = tokenize)
    for index,row in needed_data.iterrows():
        q_text = row['Q_BODY']
        a_text = row['A_BODY']
        #print(type(A),type(A))
        #print(q_text,a_text)
        q_text = removehtmltags(q_text)
        a_text = removehtmltags(a_text)
        QA_vector= tfidf_vector.fit_transform([q_text,a_text])
        #print(A.shape,Q.shape)
        Q_vector = QA_vector[0].toarray()
        A_vector = QA_vector[1].toarray()
        cs = cosine_similarity(Q_vector,A_vector)

# Sentiment Analysis
def analyzeCommentSentiment():

	col = ['READABILITY','ANSWER_SCORE','ANSWERER_SCORE','SIMILARITY','TIME_DIFF','LABEL']
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
		final_feature_df.loc[answer]  = [readability, df["A_SCORE"][answer], df["U_REPUTATION"][answer],similarity,df['TIME_DIFF'][answer],label]
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


def predictLabel():
	from sklearn.naive_bayes import BernoulliNB
	from sklearn.metrics import f1_score
	from sklearn import tree

	dtm = load_so_answers()
	# Seperate training set - Considering till last but 1000 answers
	train_data = dtm[:(dtm.shape[0] - 1000)]
	# Seperate test set - Considering last 1000 answers
	test_data = dtm[(dtm.shape[0] - 1000):]

	# Train set: Drop label column from dataset
	actual_train_data = train_data.drop('LABEL', 1)
	#actual_train_data.drop('READABILITY',axis = 1,inplace = True)
	#actual_train_data.drop('SIMILARITY',axis = 1,inplace = True)
	#actual_train_data.drop('Id',axis = 1,inplace = True)
	# Train set: Label values
	actual_train_label = train_data['LABEL'].values
	#print(actual_train_data)

	# Test set: Drop label column from dataset
	actual_test_data = test_data.drop('LABEL', 1)
	#actual_test_data.drop('READABILITY', axis= 1, inplace = True)
	#actual_test_data.drop('SIMILARITY', axis= 1, inplace = True)
	actual_test_label = train_data['LABEL'].values
	# Test set: Label values
	actual_test_label = test_data['LABEL'].values



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



	# Bernoulli classifier
	clf = BernoulliNB()

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

	clf = tree.DecisionTreeClassifier()
	clf.fit(actual_train_data, actual_train_label)
	predictions = clf.predict(actual_test_data)

	numerator = 0
	for x in range(1000):
		if predictions[x] == actual_test_label[x]:
			numerator += 1
	print ("Prediction accuracy Decision Trees... " + str(100*numerator/1000.0))
	print(f1_score(actual_test_label,predictions,average= 'binary'))

	clf = svm.SVC()
	clf.fit(actual_train_data, actual_train_label)
	predictions = clf.predict(actual_test_data)

	numerator = 0
	for x in range(1000):
		if predictions[x] == actual_test_label[x]:
			numerator += 1
	print ("Prediction accuracy SVM... " + str(100*numerator/1000.0))
	print(f1_score(actual_test_label,predictions,average= 'binary'))

def main():
	#analyzeCommentSentiment()
	predictLabel()

if __name__ == '__main__':
	main()
