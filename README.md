# 2Sigma-Rental-prediction Challenge

### Team
Abhinav Choudhury (achoudh3)

Zubin Thampi (zsthampi)

Chekad Sarami (csarami)

### Dependencies
All dependencies can be installed by running the script __install_requirements.sh__

### How to run
The primary script is main.py and takes a number of different combinations of vectorizers and classifiers as input.

    python main.py <vectorizer> <classifier>
    
\<vectorizer\> can be one of:
1. __tfidf__ - Term frequency inverse document frequency (default)
2. __lda__ - Latent Dirichlet allocation
3. __nmf__ - Use non-negative matrix factorization to vectorize the text
4. __cv__ - CountVectorizer (essentially Bag-of-words model)

\<classifier\> can be one of:
1. __xgboost__ - Gradient boosted decision trees using the XGBoost library in Python (https://github.com/dmlc/xgboost)
2. __lgb__ - Gradient boosted decision trees using Microsoft's LightGBM library in Python (https://github.com/Microsoft/LightGBM)
3. __logreg__ - Logistic regression (discarded from final use due to poor performance, only kept as a project component)
