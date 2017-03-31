import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
from dateutil.parser import parse
import datetime

### NOTES
### tfidf features = 200, eta = 0.005, maxdepth = 5. LB score ~ 0.6, train-mlogloss:0.58059, test-mlogloss:0.641615

print("Loading data...")
with open("train.json", "r") as f:
        d = json.load(f)

with open("test.json", "r") as ftest:
        dtest = json.load(ftest)

def add_date_features(d, update_df):
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

        update_df["day"] = day
        update_df["month"] = month
        update_df["year"] = year
        update_df["age"] = postingAge

def extractFeatures(dtrain, dtest, method="LDA", max_f=50, ntopics = 25):
	train_listing_id = [dtrain['listing_id'][key] for key in dtrain['listing_id']]
	test_listing_id = [dtest['listing_id'][key] for key in dtest['listing_id']]

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
                        train_cv = train_cv.toarray()
			test_cv = test_cv.toarray()
			train_df = pd.DataFrame(train_cv)
			test_df = pd.DataFrame(test_cv)
			train_df['listing_id'] = train_listing_id
			test_df['listing_id'] = test_listing_id
			train_df.sort_values(by='listing_id')
			test_df.sort_values(by='listing_id')
			return(train_df, test_df)
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
			train_cv = train_cv.toarray()
                        test_cv = test_cv.toarray()
                        train_df = pd.DataFrame(train_cv)
                        test_df = pd.DataFrame(test_cv)
                        train_df['listing_id'] = train_listing_id
                        test_df['listing_id'] = test_listing_id
                        train_df.sort_values(by='listing_id')
                        test_df.sort_values(by='listing_id')
                        return(train_df, test_df)
                else:
                        print("### Running LDA...")
                        lda = LatentDirichletAllocation(n_topics=ntopics, max_iter=5, random_state=2017)
                        train_out = lda.fit_transform(train_cv)
                        test_out = lda.transform(test_cv)
                        return(train_out, test_out)


train = pd.read_csv('train_python.csv', header=0)
test = pd.read_csv('test_python.csv', header=0)

add_date_features(d, train)
add_date_features(dtest, test)

train_features, test_features = extractFeatures(d, dtest, method="tfidf", max_f=300, ntopics = 25)
if train.shape[0] != train_features.shape[0]:
	print("train mismatch")
train = train.merge(train_features, on='listing_id')

if test.shape[0] != test_features.shape[0]:
	print("test mismatch")
test = test.merge(test_features, on='listing_id')

#train = train.drop(['longitude', 'latitude'], axis=1)
#test = test.drop(['longitude', 'latitude'], axis=1)

#train = train[(train['latitude'] >= 40.0) & (train['latitude'] <= 42.0) & (train['longitude'] <= -71.0) & (train['longitude'] >= -75.0) ]

print("Writing train and test to csv...")
train.to_csv("train_final_150_tfidf.csv", index=False)
test.to_csv("test_final_150_tfidf.csv", index=False)
print("Done.")

print("Total features are now: " + str(train.shape[1]))

def runLGB(train):
	kf = StratifiedKFold(5, random_state=2017)

	best_model = None
	min_loss = None
	for dev_idx, val_idx in kf.split(np.zeros(train.shape[0]), train['interest_level'].values):
	        train_X = train.iloc[dev_idx].drop('interest_level', axis=1).values
	        train_Y = train.iloc[dev_idx]['interest_level'].values
	        print(train_Y)
	        val_X = train.iloc[val_idx].drop('interest_level', axis=1).values
	        val_Y = train.iloc[val_idx]['interest_level'].values

	        print("### Training model...")
	        model = lgb.LGBMClassifier(boosting_type='gbdt', max_depth=6, learning_rate=0.05, n_estimators=60, subsample=0.8, colsample_bytree=0.8, drop_rate= 0.25)
	        model.fit(train_X, train_Y, eval_set=[(val_X, val_Y)], eval_metric='logloss', early_stopping_rounds=None)

	        val_preds = model.predict_proba(val_X)
	        loss = log_loss(val_Y, val_preds)

	        print("LogLoss = " + str(loss))
	        if min_loss is None or loss < min_loss:
	                min_loss = loss
	                best_model = model

	return(best_model)

kf = StratifiedKFold(5)
print("Running StratKFold...")
for dev_idx, val_idx in kf.split(np.zeros(train.shape[0]), train['interest_level']):
	train_Y = np.array(train.iloc[dev_idx]['interest_level'])
	train_X = train.iloc[dev_idx].drop('interest_level', axis=1).as_matrix()
	val_Y = np.array(train.iloc[val_idx]['interest_level'])
	val_X = train.iloc[val_idx].drop('interest_level', axis=1).as_matrix()

	preds, model = runXGB(train_X, train_Y, val_X, val_Y)
	break

test_X = test.as_matrix()
test_y = model.predict(xgb.DMatrix(test_X))

test_y_df = pd.DataFrame({})
#print(test_X["listing_id"])
test_y_df["listing_id"] = test['listing_id']
test_y_df[["low", "medium", "high"]] = pd.DataFrame(test_y)

print("Writing to CSV...")
test_y_df.to_csv("output.csv", index=False)
