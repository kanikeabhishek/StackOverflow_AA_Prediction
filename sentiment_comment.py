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


SO_DATASET = os.path.join(os.path.dirname(__file__), 'data', 'dataset.tar.gz')
SO_ANSWERS = os.path.join(os.path.dirname(__file__), 'answer.csv')

# Load Dataset
def load_so_corpus(dataset):
	return pd.read_csv(dataset)

def load_so_answers():
	return pd.read_csv(SO_ANSWERS, index_col=0)

def removehtmltags(data):
    p = re.compile(r'<.*?>')
    new_data = p.sub('', data)
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    word_tokens = word_tokenize(new_data)
    word_tokens_filtered = [w for w in word_tokens if not w in stop_words]
    bag_of_words = []
    for word in word_tokens_filtered:
        bag_of_words.append(stemmer.stem(word))
    return bag_of_words

def counter_cosine_similarity(c1, c2):
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k, 0) * c2.get(k, 0) for k in terms)
    magA = math.sqrt(sum(c1.get(k, 0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k, 0)**2 for k in terms))
    return dotprod / (magA * magB)

def calculate_cosine_similarity():
    df = load_data()
    temp= df.iloc[:,4:6]
    tfidf_vectorizer=TfidfVectorizer()
    for index,row in temp.iterrows():
        q_text = row['Q_BODY']
        a_text = row['A_BODY']
 
# Sentiment Analysis
def analyzeCommentSentiment():

	# If answer.csv is present with required number of rows return
	if load_so_answers().shape[0] == pickle.load(open("saver_pickle", "rb")).shape[0] - 1:
		return

	################################## Save stackoverflow corpus as pickle ##################################
	'''
	df = load_so_corpus(SO_DATASET)
	pickle.dump(df, open("saver_pickle", "wb"))
	'''
	################################## Save stackoverflow corpus as pickle ##################################

	# Load stackoverflow corpus from pickle
	df = pickle.load(open("saver_pickle", "rb"))

	# Answer dataframe
	answer_df = pd.DataFrame(columns = ["label", "readability", "answer_score", "User_reputation", "similarity"])

	# Readibility accuracy measure variables true: accepted variable, false: other answers varaible
	numerator_true = 0
	denominator_true = 0
	numerator_false = 0
	denominator_false = 0

	# Similarity accuracy measure variables
	sim_num = 0
	sim_deno = 0

	# Main loop which runs overall rows of corpus
	for answer in range(df.shape[0]):
		# Question ID column is saved as filename, since pandas is reading from .tar.gz (zipped) format
		if np.isnan(df["newsample-000000000000.csv"][answer]):
			continue
		question_id = int(df["newsample-000000000000.csv"][answer])

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

		question_body = df["Q_BODY"][answer]
		answer_body = df["A_BODY"][answer]
		question_body = removehtmltags(question_body)
		answer_body = removehtmltags(answer_body)
		counter_q_text = Counter(question_body)
		counter_a_text = Counter(answer_body)
		similarity = counter_cosine_similarity(counter_q_text, counter_a_text)
		answer_df.loc[answer] = [label, readability, df["A_SCORE"][answer], df["U_REPUTATION"][answer], similarity]
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

	answer_df.to_csv("answer.csv")

def predictLabel():
	from sklearn.naive_bayes import BernoulliNB

	dtm = load_so_answers()
	# Seperate training set - Considering till last but 1000 answers
	train_data = dtm[:(dtm.shape[0] - 1000)]
	# Seperate test set - Considering last 1000 answers
	test_data = dtm[(dtm.shape[0] - 1000):]

	# Train set: Drop label column from dataset
	actual_train_data = train_data.drop('label', 1)
	# Train set: Label values
	actual_train_label = train_data['label'].values

	# Test set: Drop label column from dataset
	actual_test_data = test_data.drop('label', 1)
	# Test set: Label values
	actual_test_label = test_data['label'].values

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
	print ("Prediction accuracy... " + str(100*numerator/1000.0))

def main():
	analyzeCommentSentiment()
	predictLabel()

if __name__ == '__main__':
	main()


