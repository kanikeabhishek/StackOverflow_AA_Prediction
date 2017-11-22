import pandas as pd
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
from textblob import TextBlob
import xml.etree.ElementTree as ET
from textstat.textstat import textstat

SO_DATASET = os.path.join(os.path.dirname(__file__), 'data', 'dataset.tar.gz')

# Load Dataset
def load_so_corpus(dataset):
	return pd.read_csv(dataset)

# Sentiment Analysis
def analyzeCommentSentiment():
	# df = load_so_corpus(SO_DATASET)
	# pickle.dump(df, open("saver_pickle", "wb"))
	df = pickle.load(open("saver_pickle", "rb"))

	answer_df = pd.DataFrame(columns = ["label"])
	index = 0
	for answer in range(index, df.shape[0]):
		label = (1 if df["A_ID"][answer] == df["Accepted_Answer_ID"][answer] else 0)
		# answer_df.append({"label": label})
		print ("adding label at " + str(answer) + " label " + str(label))
		answer_df.loc[answer] = [label]
	# pd.write_csv(answer_df, )
	answer_df.to_csv("answer.csv")
def main():
	analyzeCommentSentiment()

if __name__ == '__main__':
	main()


