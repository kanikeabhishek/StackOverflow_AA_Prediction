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

# Load Dataset
def load_so_corpus(dataset):
	return pd.read_csv(dataset)

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
    #answer_body = df['A_BODY']
    for index,row in temp.iterrows():
        q_text = row['Q_BODY']
        a_text = row['A_BODY']
 
# Sentiment Analysis
def analyzeCommentSentiment():
	# df = load_so_corpus(SO_DATASET)
	# pickle.dump(df, open("saver_pickle", "wb"))
	df = pickle.load(open("saver_pickle", "rb"))

	answer_df = pd.DataFrame(columns = ["label", "readability", "answer_score", "User_reputation", "similarity"])
	index = 0
	numerator_true = 0
	denominator_true = 0
	numerator_false = 0
	denominator_false = 0
	sim_num = 0
	sim_deno = 0
	for answer in range(index, df.shape[0]):
		# print ("Proccessed: " + str(answer))
		if np.isnan(df["newsample-000000000000.csv"][answer]):
			continue
		question_id = int(df["newsample-000000000000.csv"][answer])

		label = (1 if df["A_ID"][answer] == df["Accepted_Answer_ID"][answer] else 0)

		answer_body = df["A_BODY"][answer]
		soup = BeautifulSoup(answer_body, "lxml")

		try:
			if soup.find('code'):
				for _ in range(len(soup.find_all('code'))):
					if soup.find('code'):
						soup.find('code').replaceWith('A good working code')
					else:
						break

			if soup.find('a'):
				for _ in range(len(soup.find_all('a'))):
					if soup.find('a'):
						soup.find('a').replaceWith('A working link')
					else:
						break
		except Exception as e:
			print ("answer: " + df["A_BODY"][answer])
			raise e

		# for sentence in soup.find_all('p'):
		# 	if sentence.find('a'):
		# 		for code_sentence in range(len(sentence.find_all('a'))):
		# 			sentence.find_all('a')[0].decompose()

		# answer_body = answer_body + sentence.text

		if soup != "":
			readability = textstat.flesch_reading_ease(soup.text)
		else:
			if label:
				readability = 100.0
			else:
				readability = 0.0

		# if (readability <= 0 and label == 1):
		# 	print ("adding label at " + str(answer) + " label " + str(label))
		# 	print ("Question ID: " + str(int(df["newsample-000000000000.csv"][answer])))
		# 	print ("read: " + str(readability))
		# 	# print ("answer: " + df["A_BODY"][answer])
		# 	# print ("soup: " + str(soup))
		# 	print ("parsed answer: " + soup.text)
		# 	print ("-"*100 + "\n\n")

		# Similarity
		question_body = df["Q_BODY"][answer]
		answer_body = df["A_BODY"][answer]
		question_body = removehtmltags(question_body)
		answer_body = removehtmltags(answer_body)
		counter_q_text = Counter(question_body)
		counter_a_text = Counter(answer_body)
		similarity = counter_cosine_similarity(counter_q_text, counter_a_text)
		answer_df.loc[answer] = [label, readability, df["A_SCORE"][answer], df["U_REPUTATION"][answer], similarity]

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

	# pd.write_csv(answer_df, )
	answer_df.to_csv("answer.csv")
def main():
	analyzeCommentSentiment()

if __name__ == '__main__':
	main()


