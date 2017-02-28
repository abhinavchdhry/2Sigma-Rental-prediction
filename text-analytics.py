import json
import nltk
import pandas as pd
import numpy as np
import operator

with open("train.json", "r") as f:
	d = json.load(f)

price = [d["price"][key] for key in d["price"]]
bedrooms = [d["bedrooms"][key] for key in d["bedrooms"]]
bathrooms = [d["bathrooms"][key] for key in d["bathrooms"]]
images = d["photos"]
images = [len(images[key]) for key in images]
description = [d["description"][key] for key in d["description"]]
description = [i.encode('ascii', 'ignore') for i in description]
description = ["".join([ch for ch in article if (ch.isalnum() or ch.isspace())]) for article in description]    # Description content filtered off non-alphanum chars

interest = d["interest_level"]
interest = [interest[key] for key in interest]

df = {"price" : price, "bedrooms":bedrooms, "bathrooms":bathrooms, "images":images, "description":description, "interest":interest }
df = pd.DataFrame(df)

print("Data frame created...")

desc_high = list(df[df["interest"]=="high"]["description"])
desc_med = list(df[df["interest"]=="medium"]["description"])
desc_low = list(df[df["interest"]=="low"]["description"])


def genWordPercentChart(description):
	stopwords = set(["and", "in", "a", "the", "of", "to", "with", "an", "br", "is", "this", "for", "websiteredacted"])
	dict_freq = {}
	N = len(description)
	for desc in description:
		words = desc.split()
		words = [word.strip().lower() for word in words]
		words = set(words)
		words = words - stopwords	# remove stopwords
		for word in words:
			if word in dict_freq:
				dict_freq[word] += 1
			else:
				dict_freq[word] = 1
	for key in dict_freq:
		dict_freq[key] = float(dict_freq[key])/float(N)

	return(dict_freq)

print("Creating dictionaries...")
dict_high = genWordPercentChart(desc_high)
dict_med = genWordPercentChart(desc_med)
dict_low = genWordPercentChart(desc_low)

sorted_high = sorted(dict_high.items(), key=operator.itemgetter(1), reverse=True)
sorted_med = sorted(dict_med.items(), key=operator.itemgetter(1), reverse=True)
sorted_low = sorted(dict_low.items(), key=operator.itemgetter(1), reverse=True)

print("Filtering unique keys per class...")
common_words = set(dict_high.keys()).intersection(set(dict_med.keys())).intersection(set(dict_low.keys()))
sorted_high = [tup for tup in sorted_high if tup[0] in common_words]
sorted_med = [tup for tup in sorted_med if tup[0] in common_words]
sorted_low = [tup for tup in sorted_low if tup[0] in common_words]

print(sorted_high[0:10])
print(sorted_med[0:10])
print(sorted_low[0:10])

print(len(common_words))

# The real stuff: We want to find distinctive common words
# i.e. words that are common to all 3 classes but have varying distributions among the classes
# like if a word is present among "High" classes 20% of the time, in "medium" classes 50% of the time and in "Low classes" 90% of the time, the word is a distinguishing factor
# as opposed to a word which has nearly equal percentages in all 3 classes
# Even better distinguisher would be a word that has similar distribution in one class,
# and widely different distribution in the third class
# We want to find such words
print("Determining keywords...")
threshold = 0.05
keywords = []
maxdiff = 0
for word in common_words:
	freqs = [dict_high[word], dict_med[word], dict_low[word]]
	freqs.sort()
	if (freqs[2] - freqs[0] >= threshold):
		keywords.append(word)
	if (freqs[2] - freqs[0] > maxdiff):
		maxdiff = freqs[2] - freqs[0]
print(len(keywords))
print(maxdiff)
