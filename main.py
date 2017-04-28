import sys
import pandas as pd
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import json, math
from random import sample
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from dateutil.parser import parse
import xgboost as xgb
from sklearn import model_selection
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
import datetime
from itertools import product
import lightgbm as lgb

### Argument check
if (len(sys.argv) != 3):
	print("""Usage: python main.py <text-vectorizer> <classifier>
		--- classifier: can be one of [lgb, xgboost, logreg]
		--- text-vectorizer: one of [cv, tfidf, nmf, LDA]""")
	exit(0)

_enableDV = 0
_enableLoc = 0
_classifier = sys.argv[2]
_textvectorizer = sys.argv[1]


### Different classifiers to try out
classifiers = {
	"logreg" : lr(),
	"logregl2" : lr(penalty='l2', solver='sag'),
	"rfc" : rfc(n_estimators=20),
	"nn" : MLPClassifier(hidden_layer_sizes=(100,100,)),
	"nb" : GaussianNB(),
	"qda" : QuadraticDiscriminantAnalysis(),
	"adboost" : AdaBoostClassifier()
	}

########### List of function definitions, includes both used and experimental functions ###########

### PART 1: Extract Doc2Vec features and represent description using that feature vector [UNUSED/EXPERIMENTAL]
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
	return(df, model)


### PART 2: Merge the location data [UNUSED/EXPERIMENTAL]
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

if _enableLoc:
	print("Reading the location data...")
	ldf = mergeLocationData()
	print(ldf.head(n=3))

### Extracts attributes from "features" vector and "description" using one of the
### following vectorization methods: tfidf, LDA, NMF, cv (countvectorizer/BoW)
### Description is preprocessed by removing all non-alphabetic character
### Elements in "features" vector are joined into one sentence
### Finally the preprocessed "description" and the sentence are merged into a text blob
### and sent to the vectorizer algorithm
def extractTextFeatures(dtrain, dtest, method="tfidf", max_f=50, ntopics = 25):
	train_features = [" ".join(dtrain["features"][key]) for key in dtrain["features"]]
	test_features = [" ".join(dtest["features"][key]) for key in dtest["features"]]

	print("### Analyzing description...")
        description = dtrain["description"]
        description = [description[key].encode('ascii', 'ignore') for key in description]
        train_desc = ["".join([ch for ch in article if (ch.isalnum() or ch.isspace())]) for article in description]    # Description content filtered off non-alphanum chars

	description = dtest["description"]
        description = [description[key].encode('ascii', 'ignore') for key in description]
        test_desc = ["".join([ch for ch in article if (ch.isalnum() or ch.isspace())]) for article in description]    # Description content filtered off non-alphanum chars

	train_features = [train_features[i] + " " + train_desc[i] for i in range(0, len(train_features))]
	test_features = [test_features[i] + " " + test_desc[i] for i in range(0, len(test_features))]

	if method == "tfidf" or method == "nmf":
		print("### Running TFIDF...")
		tfidf = TfidfVectorizer(stop_words='english', max_features=max_f, ngram_range=(2, 2))
		train_cv = tfidf.fit_transform(train_features)
		test_cv = tfidf.transform(test_features)
		if method == "tfidf":
			return(train_cv.toarray(), test_cv.toarray())
		else:
			print("### Running NMF...")
			nmf = NMF(n_components=ntopics, random_state=2017, alpha=0.1, verbose=1)
			train_out = nmf.fit_transform(train_cv)
			test_out = nmf.transform(test_cv)
			return(train_out, test_out)
	
	if method == "cv" or method == "LDA":
		print("### Running CountVectorizer...")
		cv = CountVectorizer(max_features=max_f, stop_words='english', ngram_range=(2,2))
		train_cv = cv.fit_transform(train_features)
		test_cv = cv.transform(test_features)
		if method == "cv":
			return(train_cv.toarray(), test_cv.toarray())
		else:
			print("### Running LDA...")
			lda = LatentDirichletAllocation(n_topics=ntopics, max_iter=5, random_state=2017)
			train_out = lda.fit_transform(train_cv)
			test_out = lda.transform(test_cv)
			return(train_out, test_out)


### Extracts the base attributes from the given dataset and their trivial combinations
### Weekday, Month, Year, and Age are extracted from Date string
### High-cardinality categorical variables are converted to integers using LabelEncoder()
### For test data, original listing_ids are returned
def appendBaseFeatures(d, df, type):
	price = [d["price"][key] for key in d["price"]]
	bedrooms = [d["bedrooms"][key] for key in d["bedrooms"]]
	bathrooms = [d["bathrooms"][key] for key in d["bathrooms"]]
	images = d["photos"]
	images = [len(images[key]) for key in images]
	building_id = [d["building_id"][key] for key in d["building_id"]]
        manager_id = [d["manager_id"][key] for key in d["manager_id"]]
       	listing_id = [d["listing_id"][key] for key in d["listing_id"]]
        latitude = [d["latitude"][key] for key in d["latitude"]]
        longitude = [d["longitude"][key] for key in d["longitude"]]
	display_address = [d["display_address"][key] for key in d["display_address"]]
	street_address = [d["street_address"][key] for key in d["street_address"]]

	if (type == 'train'):
		interest = d["interest_level"]
		interest = [interest[key] for key in interest]

	df["price"] = price
	df["bedrooms"] = bedrooms
	df["bathrooms"] = bathrooms
	df["images"] = images

	# Categorical attributes
	lblr = LabelEncoder()
	lblr.fit(building_id)
	df["building_id"] = lblr.transform(building_id)

	lblr = LabelEncoder()
	lblr.fit(manager_id)
	df["manager_id"] = lblr.transform(manager_id)

	lblr = LabelEncoder()
	lblr.fit(listing_id)
	df["listing_id"] = lblr.transform(listing_id)

	lblr = LabelEncoder()
	lblr.fit(display_address)
	df["display_address"] = lblr.transform(display_address)

	lblr = LabelEncoder()
	lblr.fit(street_address)
	df["street_address"] = lblr.transform(street_address)

	df["latitude"] = latitude
	df["longitude"] = longitude

	df["pricePerBed"] = df['price'] / df['bedrooms']
	df["pricePerBath"] = df['price'] / df['bathrooms']
	df["pricePerRoom"] = df['price'] / (df['bedrooms'] + df['bathrooms'])
	df["bedPerBath"] = df['bedrooms'] / df['bathrooms']
	df["bedBathDiff"] = df['bedrooms'] - df['bathrooms']
	df["bedBathSum"] = df["bedrooms"] + df['bathrooms']
	df["bedsPerc"] = df["bedrooms"] / (df['bedrooms'] + df['bathrooms'])

	if type == 'train':
		df["interest"] = interest

	day = []
	month = []
	year = []
	postingAge = []
	created = [d["created"][key] for key in d["created"]]
	for date in created:
		b = parse(date)
		day.append(b.weekday())
		month.append(b.month)
		year.append(b.year)
		now = datetime.datetime.now()
		postingAge.append((now-b).days)

	df["day"] = day
	df["month"] = month
	df["year"] = year
	df["age"] = postingAge

	# Return the original listing_id strings. Need it for test set .csv listing_id column
	return(listing_id)


def factorize(df1, df2, column):
    ps = df1[column].append(df2[column])
    factors = pd.factorize(ps)[0]
    df1[column] = factors[:len(df1)]
    df2[column] = factors[len(df1):]
    return df1, df2


"""
See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
Classification and Prediction Problems" by Daniele Micci-Barreca
"""
def hcc_encode(train_df, test_df, variable, target, prior_prob, k, f=1, g=1, r_k=None, update_df=None):
    hcc_name = "_".join(["hcc", variable, target])

    grouped = train_df.groupby(variable)[target].agg({"size": "size", "mean": "mean"})
    grouped["lambda"] = 1 / (g + np.exp((k - grouped["size"]) / f))
    grouped[hcc_name] = grouped["lambda"] * grouped["mean"] + (1 - grouped["lambda"]) * prior_prob

    df = test_df[[variable]].join(grouped, on=variable, how="left")[hcc_name].fillna(prior_prob)
    if r_k: df *= np.random.uniform(1 - r_k, 1 + r_k, len(test_df))     # Add uniform noise. Not mentioned in original paper

    if update_df is None: update_df = test_df
    if hcc_name not in update_df.columns: update_df[hcc_name] = np.nan
    update_df.update(df)
    return


### Remove outliers (lat, long) from the training data. In this case, whatever (lat, long) values lie far away from NY
### Some initial tests and plotting suggest valid data should have:
### 40.0 <= latitude <= 42.0
### -75.0 <= longitude <= -71.0 
def filterOutliers(df, dftest):
	df = df[(df['latitude'] >= 40.0) & (df['latitude'] <= 42.0) & (df['longitude'] <= -71.0) & (df['longitude'] >= -75.0) ]
	return(df, dftest)


# Feature selection pipeline [UNUSED]
def selectFeatures(dftrain, dftest):
	trainY = dftrain["interest"]
	trainX = dftrain.drop("interest", axis=1).as_matrix()
	test = dftest.as_matrix()

	k = int(round(trainX.shape[1] * 0.75))
#	model = SelectKBest(chi2, k)
#	train_new = model.fit_transform(trainX, trainY)
#	test_new = model.transform(test)

#	clf = ExtraTreesClassifier()
	print("RFC")
	clf = rfc(n_estimators = 10)
	clf = clf.fit(trainX, trainY)
	print("Swelect")
	model = SelectFromModel(clf, prefit=True)

	train_new = pd.DataFrame(model.transform(trainX))
	print("Intermediate shape:")
	print(train_new.shape)
	print(trainY.shape)
#	train_new = train_new.join(pd.Series(trainY, index=train_new.index))
	train_new = train_new.assign(interest=pd.Series(trainY).values) 
	test_new = pd.DataFrame(model.transform(test))
	
	print("Out shape:")
	print(train_new.shape)
	print(test_new.shape)
	return(train_new, test_new)

### XGBoost model
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=200):
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.05
	param['max_depth'] = 5
	param['silent'] = 1
	param['num_class'] = 3
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 1
	param['subsample'] = 0.7
	param['colsample_bytree'] = 0.7
	param['seed'] = seed_val
	num_rounds = num_rounds

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)

	if test_y is not None:
		xgtest = xgb.DMatrix(test_X, label=test_y)
		watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
		model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
	else:
		xgtest = xgb.DMatrix(test_X)
		watchlist = [ (xgtrain, 'train') ]
		model = xgb.train(plst, xgtrain, num_rounds, watchlist)

	pred_test_y = model.predict(xgtest)
	return pred_test_y, model

### Create a model based on _classifier
def createAndRunModel(train_X, train_y, test_X, test_y=None):
	print("## Training " + _classifier + " model...")
	if _classifier == "xgboost":
		preds, model = runXGB(train_X, train_y, test_X, test_y)

	elif _classifier == "lgb":
		model = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_leaves=60, max_depth=6, learning_rate=0.01, n_estimators=1000, subsample=1, colsample_bytree=1, reg_lambda=0)
                model.fit(train_X, train_y, eval_set=[(test_X, test_y)], eval_metric='multi_logloss', early_stopping_rounds=20)
                preds = model.predict_proba(test_X)

	elif _classifier == "logreg":
		train_df = pd.DataFrame(train_X)
		train_df["train_Y"] = pd.Series(train_y, index=train_df.index)
		train_df = train_df.replace([np.inf, -np.inf], np.nan).dropna(how='all').fillna(0)

		train_Y = train_df["train_Y"].values
		train_X = train_df.drop("train_Y", axis=1).values
		model = lr()
		model.fit(train_X, train_y)
		
		test_X = pd.DataFrame(test_X).replace([np.inf, -np.inf], np.nan).fillna(0).values
		preds = model.predict_proba(test_X)

	else:
		model = classifiers[_classifier]
		preds = None

	return(preds, model)


##### CODE #####
## Construct the full DataFrame
print("### Loading data from JSON files...")
with open("train.json", "r") as f:
        d = json.load(f)

with open("test.json", "r") as ftest:
	dtest = json.load(ftest)

# Initialize empty data frames for training and test data
df = pd.DataFrame({})
dftest = pd.DataFrame({})

# Append the base features to the train and test dataframes
appendBaseFeatures(d, df, 'train')
origListingIds = appendBaseFeatures(dtest, dftest, 'test')

### Append the extracted features
print("### Extracting text features, using " + _textvectorizer + "...")
train_features, test_features = extractTextFeatures(d, dtest, method=_textvectorizer, max_f=250, ntopics=100)
train_fcols = ["train_fcol_" + str(i) for i in range(0, train_features.shape[1])]
test_fcols = ["test_fcol_" + str(i) for i in range(0, test_features.shape[1])]

df_tr_f = pd.DataFrame(train_features)
df_tr_f.columns = train_fcols

df_te_f = pd.DataFrame(test_features)
df_te_f.columns = test_fcols

df = df.join(df_tr_f)
dftest = dftest.join(df_te_f)

### PREPROCESSING
print("### Preprocessing...")
df, dftest = filterOutliers(df, dftest)

for attr in ['manager_id', 'building_id', 'display_address', 'street_address']:
	factorize(df, dftest, attr)

N = df.shape[0]
iters = 5

print("### Model training and cross validation...")

df = df.replace({"interest": {"low": 0, "medium": 1, "high": 2}})
df = df.join(pd.get_dummies(df["interest"], prefix="pred").astype(int))
prior_0, prior_1, prior_2 = df[["pred_0", "pred_1", "pred_2"]].mean()
attributes = product(("building_id", "manager_id"), zip(("pred_1", "pred_2"), (prior_1, prior_2)))

### Training model
print("### Running cross-validation...")
cv_scores = []
kf = model_selection.StratifiedKFold(5)
best_model = None
best_score = 1.0
count = 0
train_df = df
test_df = dftest
for dev_index, val_index in kf.split(np.zeros(df.shape[0]), df['interest']):
	df = train_df
	dftest = test_df
	print("## Iteration: " + str(count+1))
	for variable, (target, prior) in attributes:
		hcc_encode(df.iloc[dev_index], df.iloc[val_index], variable, target, prior, k=5, r_k=None, update_df=df)
		hcc_encode(df, dftest, variable, target, prior, k=5, r_k=None, update_df=None)
	df = df.drop(['pred_0', 'pred_1', 'pred_2', 'manager_id', 'building_id'], axis=1)
	dftest = dftest.drop(['manager_id', 'building_id'], axis=1)
	train_XY = df.iloc[dev_index]
	val_Y = df.iloc[val_index]['interest']
	val_X = df.iloc[val_index].drop('interest', axis=1)
	test_X = dftest
	train_Y = train_XY['interest']
	train_X = train_XY.drop('interest', axis=1)
	train_X = train_X.as_matrix()
	train_Y = np.array(train_Y)
	val_X = val_X.as_matrix()
	val_Y = np.array(val_Y)

	preds, model = createAndRunModel(train_X, train_Y, val_X, val_Y)
	if best_model is None or log_loss(val_Y, preds) < best_score:
		best_score = log_loss(val_Y, preds)
		best_model = model
	count += 1
	print("## log_loss score: " + str(log_loss(val_Y, preds)))

print("### Best model has log_loss: " + str(best_score))

### Do not work on test data if not using lgb or xgboost
if _classifier in ["xgboost", "lgb"]:
	test_X = test_X.as_matrix()
	if _classifier == "xgboost":
		test_y = model.predict(xgb.DMatrix(test_X))
	else:
		test_y = model.predict_proba(test_X)
	test_y_df = pd.DataFrame({})
	test_y_df["listing_id"] = origListingIds
	test_y_df[["high", "medium", "low"]] = pd.DataFrame(test_y)
	print("### Writing output to CSV...")
	test_y_df.to_csv("output.csv", index=False)
	print("\nDone. Output written to 'output.csv'. Upload this file to competition submission page to get test log-loss.")
else:
	print("\nDone. Note that test predictions are not written if model is not lgb or xgboost.")

## TEST REPORTS
## Public leaderboard LogLoss ~ 0.7056 with early stopping at 100 iterations, max_depth = 7, tfidf n_features = 250
## Public leaderboard LogLoss ~ 0.6719 with early stopping at 50 iterations, max_depth = 7, tfidf n_features = 250
## LogLoss ~ 0.68 training on full data, 50 iters, max_depth = 7, """
## LogLoss ~ 0.71482 full training data, 75 iterations, training mlogloss ~ 0.47  [ Overfit ]
## LogLoss ~ 0.66735 full training data, 25 iterations, training mlogloss ~ 0.574
