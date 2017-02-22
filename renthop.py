import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import json
from random import sample
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import log_loss

print("Loading data...")
with open("train.json", "r") as f:
	d = json.load(f)

### PART 1: Extract Doc2Vec features and represent description using that feature vector

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


## PART 2: Construct the full DataFrame
price = [d["price"][key] for key in d["price"]]
bedrooms = [d["bedrooms"][key] for key in d["bedrooms"]]
bathrooms = [d["bathrooms"][key] for key in d["bathrooms"]]

interest = d["interest_level"]
interest = [interest[key] for key in interest]

df["price"] = price
df["bedrooms"] = bedrooms
df["bathrooms"] = bathrooms
df["interest"] = interest

#descVsInterest = {"desc" : description, "interest" : interest }
#descVsInterest = pd.DataFrame(descVsInterest)

N = len(description)
iters = 5

print("### Model training and cross validation...")
for i in range(iters):
	print("Iteration " + str(i) + "...")
	test_indices = sample(range(0, N), 10000)
	train_indices = list(set(range(0, N)) - set(test_indices))

	train = df.iloc[train_indices]
	test = df.iloc[test_indices]

	model = lr()	# Logistic regression model
	cols = list(df)
	cols.remove("interest")

	print("Fitting model...")
	model.fit(train[cols], np.reshape(train["interest"], -1))

	actual = np.reshape(test["interest"], -1)
	predicted = model.predict_proba(test[cols])

	print(pd.DataFrame(predicted).head(n=3))

	loss = log_loss(actual, predicted, labels=model.classes_)
	print("Log loss: " + str(loss))
