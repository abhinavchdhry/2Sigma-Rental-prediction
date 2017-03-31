import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import sys, random
from sklearn.metrics import log_loss

print("### Loading data...")
train = pd.read_csv('train_final_150_tfidf.csv', header=0)
test = pd.read_csv('test_final_150_tfidf.csv', header=0)

#trainY = train['interest_level']
#trainX = train.drop('interest_level', axis=1)

# Last best setting: num_leaves = 60, max_depth = 6, eta = 0.01, n_esti = 1000, early_stopping=10

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
		model = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_leaves=60, max_depth=6, learning_rate=0.01, n_estimators=1000, subsample=1, colsample_bytree=1, reg_lambda=0)
		model.fit(train_X, train_Y, eval_set=[(val_X, val_Y)], eval_metric='multi_logloss', early_stopping_rounds=20)
	
		val_preds = model.predict_proba(val_X)
		loss = log_loss(val_Y, val_preds)

		print("LogLoss = " + str(loss))
		if min_loss is None or loss < min_loss:
			min_loss = loss
			best_model = model

	print("Min loss is: " + str(min_loss))
	return(best_model)

best_model = runLGB(train)
test_y = best_model.predict_proba(test.values)

test_y_df = pd.DataFrame({})
#print(test_X["listing_id"])
test_y_df["listing_id"] = test['listing_id']
test_y_df[["low", "medium", "high"]] = pd.DataFrame(test_y)

print("Writing to CSV...")
test_y_df.to_csv("output.csv", index=False)
