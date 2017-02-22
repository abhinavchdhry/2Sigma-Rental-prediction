import sys
import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import json
from random import sample
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#_testcase = sys.argv[1]

print("Loading data...")
with open("train.json", "r") as f:
	d = json.load(f)

# Different classifiers to try out
classifiers = [
	{"LogRegression" : lr()},
	{"LogRegressionL2" : lr(penalty='l2', solver='sag')},
	{"RFC20" : rfc(n_estimators=20)},
	{"MLPClassifier" : MLPClassifier(hidden_layer_sizes=(100,100,))},
	{"GaussianNB" : GaussianNB()},
	{"QDA" : QuadraticDiscriminantAnalysis()},
	{"Adaboost" : AdaBoostClassifier()}
	]

### PART 1: Extract Doc2Vec features and represent description using that feature vector
def desc2vector(d):
	description = d["description"]
	description = [description[key].encode('ascii', 'ignore') for key in description]
	description = ["".join([ch for ch in article if (ch.isalnum() or ch.isspace())]) for article in description]	# Description content filtered off non-alphanum chars

	tagged = []
	for i in range(0, len(description)):
		words = [word.strip() for word in description[i].split() if word.strip() != ""]
		t = TaggedDocument(words=words, tags=["SENT_" + str(i)])
		tagged.append(t)

	# Train the Doc2Vec model
	numFeatures = 100
	model = Doc2Vec(size=numFeatures, alpha= 0.025, min_alpha=0.025, window=8, min_count=5, workers=4)
	print("Building vocab...")
	model.build_vocab(tagged)
	for epoch in range(5):
		print("Training epoch " + str(epoch) + "...")
		model.train(tagged)
		model.alpha -= 0.002
		model.min_alpha = model.alpha

	labels = ["SENT_" + str(i) for i in range(len(description))]
	vecs = [list(i) for i in model.docvecs[labels]]

	# Create a DataFrame of the vector representation of the description
	df = pd.DataFrame(vecs, columns = ["F_" + str(i) for i in range(numFeatures)])
	return(df)


### PART 2: Merge the location data
def mergeLocationData():
	locationf = open("complete.json", "r")
	data = []
	for line in locationf:
		entry = json.loads(line)
		_inner = entry.values()[0]
		temp = []
		for key in _inner:
			temp.append(0 if _inner[key] is None else _inner[key])
		data.append(temp)

	df = pd.DataFrame(data, columns=["unrated_bars", "dist2LAG", "unrated_clubs", "nearest_uni", "clubs_nearby", "nearest_store", "dist2JFK", "avg_club_rating", "nearest_mall", "avg_bar_rating", "dist2NAL", "bars_nearby", "nearest_transit"])
	return(df)

print("Reading the location data...")
ldf = mergeLocationData()
print(ldf.head(n=3))

## PART 3: Construct the full DataFrame
df = desc2vector(d)
df[list(ldf)] = ldf

print("Data frames merged...")

price = [d["price"][key] for key in d["price"]]
bedrooms = [d["bedrooms"][key] for key in d["bedrooms"]]
bathrooms = [d["bathrooms"][key] for key in d["bathrooms"]]
images = d["photos"]
images = [len(images[key]) for key in images]

interest = d["interest_level"]
interest = [interest[key] for key in interest]

#df = {"price" : price, "bedrooms":bedrooms, "bathrooms":bathrooms, "images":images, "interest":interest }
#df = pd.DataFrame(df)
df["price"] = price
df["bedrooms"] = bedrooms
df["bathrooms"] = bathrooms
df["images"] = images
df["interest"] = interest

#descVsInterest = {"desc" : description, "interest" : interest }
#descVsInterest = pd.DataFrame(descVsInterest)

N = df.shape[0]
iters = 5

print("### Model training and cross validation...")
for _model in classifiers:
	_name = _model.keys()[0]
	print("Current model: " + _name)
	model = _model[_name]
	log_loss_sum = 0
	for i in range(iters):
		print("Iteration " + str(i) + "...")
		test_indices = sample(range(0, N), 10000)
		train_indices = list(set(range(0, N)) - set(test_indices))

		train = df.iloc[train_indices]
		test = df.iloc[test_indices]

#		model = lr()	# Logistic regression model
		cols = list(df)
		cols.remove("interest")

		print("Fitting model...")
		model.fit(train[cols], np.reshape(train["interest"], -1))

		actual = np.reshape(test["interest"], -1)
		predicted = model.predict_proba(test[cols])

#		print(pd.DataFrame(predicted).head(n=3))

		loss = log_loss(actual, predicted, labels=model.classes_)
		log_loss_sum += loss
		print("Log loss: " + str(loss))

	print("Avg. log loss for " + _name + " classifier is: " + str(log_loss_sum/float(iters)))
