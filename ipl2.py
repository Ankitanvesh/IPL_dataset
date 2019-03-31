import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import re
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import skew
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5
RMSE = make_scorer(mean_squared_error_, greater_is_better=False)    
    
def create_submission(prediction,score):
    now = datetime.datetime.now()
    sub_file = 'submission_'+str(score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test['Game ID'].values, 'Winner (team 1=1, team 2=0)': prediction}).to_csv(sub_file, index=False)

def data_preprocess(train,test):
    train["Winner (team 1=1, team 2=0)"] = np.log1p(train["Winner (team 1=1, team 2=0)"])
    y = train["Winner (team 1=1, team 2=0)"]
    train=train.drop(['Game ID','Team 1','Team 2','Winner (team 1=1, team 2=0)','City','DayOfWeek','DateOfGame','TimeOfGame'], axis=1)
    print(train.head())
    test=test.drop(['Game ID','Team 1','Team 2','DateOfGame','TimeOfGame','CityOfGame', 'Day','Winner (team 1=1, team 2=0)'], axis=1)
    print(train.head())
    all_data = pd.concat([train,test],sort=False)
    print(all_data.head())
  
    
    #log transform skewed numeric features
    print("*********************")

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    print(numeric_feats)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index
    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    all_data = pd.get_dummies(all_data)
    all_data = all_data.fillna(all_data.mean())
    X_train = all_data[:train.shape[0]]
    X_test = all_data[train.shape[0]:]
    

    return X_train,X_test,y
    
def model_random_forecast(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    rfr = RandomForestRegressor(n_jobs=1, random_state=0)
    param_grid = {'n_estimators': [500], 'max_features': [5,10,15], 'max_depth':[3,5,7,9,11]}
    model = GridSearchCV(estimator=rfr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Random forecast regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_gradient_boosting_tree(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    gbr = GradientBoostingRegressor(random_state=0)
    param_grid = {
        'n_estimators': [500],
        'max_features': [10,15],
	'max_depth': [6,8,10],
       'learning_rate': [0.05,0.1,0.15,0.01],
        'subsample': [0.8]
    }
    model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Gradient boosted tree regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_xgb_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain 
    
    xgbreg = xgb.XGBRegressor(seed=0)
    param_grid = {
        'n_estimators': [500],
        'learning_rate': [ 0.01,0.07,0.1,0.2,0.05],
        'max_depth': [ 7, 9, 6,8,11],
      'subsample': [ 0.8],
      'colsample_bytree': [0.75,0.8,0.85],
    }
    model = GridSearchCV(estimator=xgbreg, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('eXtreme Gradient Boosting regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_

def model_extra_trees_regression(Xtrain,Xtest,ytrain):
    
    X_train = Xtrain
    y_train = ytrain
    
    etr = ExtraTreesRegressor(n_jobs=1, random_state=0)
    param_grid = {'n_estimators': [500], 'max_features': [10,15,20]}
    model = GridSearchCV(estimator=etr, param_grid=param_grid, n_jobs=1, cv=10, scoring=RMSE)
    model.fit(X_train, y_train)
    print('Extra trees regression...')
    print('Best Params:')
    print(model.best_params_)
    print('Best CV Score:')
    print(-model.best_score_)

    y_pred = model.predict(Xtest)
    return y_pred, -model.best_score_


# read data, build model and do prediction
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in train.columns.values]

test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in test.columns.values]
Xtrain,Xtest,ytrain = data_preprocess(train,test)


#test_predict,score = model_random_forecast(Xtrain,Xtest,ytrain)
#test_predict,score = model_xgb_regression(Xtrain,Xtest,ytrain)
#test_predict,score = model_extra_trees_regression(Xtrain,Xtest,ytrain)
test_predict,score = model_gradient_boosting_tree(Xtrain,Xtest,ytrain)

create_submission(np.exp(test_predict),score)