import pandas as pd
import numpy as np
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer

SO_DATASET = os.path.join(os.path.dirname(__file__), 'data', 'dataset.csv.tar.gz')

# Load Dataset
def load_so_corpus(dataset):
	return pd.read_csv(dataset)

# Sentiment Analysis
def analyzeCommentSentiment():
	df = load_so_corpus(SO_DATASET)
	# Sentiment Analyzer based on vader dataset
	sid = SentimentIntensityAnalyzer()
	# Top 10 comments polarity. Check ['compound'] property for polarity.
	# -1 to 0 Negative polarity, 0 to +1 Positive polarity.
	index = 0
	for comment in range(index, index+10):
		print ("\nComment:\n" + df["C_TEXT"][comment])
		print (sid.polarity_scores(df["C_TEXT"][comment]))

def main():
	analyzeCommentSentiment()

if __name__ == '__main__':
	main()
