# IPL_dataset
The Indian Premier League (IPL) is a professional league for Twenty20 cricket championship in India. It started in the year 2007-2008 and is running in its ninth year of existence. The pattern of the league is such that each team plays all other teams twice in the league stage, one in the HOME venue and other in the AWAY venue. After the league stage top four teams enter the semi-final stage and the top two teams enter the final contest.

## Objective
The goal of the contest is to develop a model to predict likelihood of a team winning the match. The true label of which team (team batting first or second) won the match is provided as “Team Won”. See data dictionary for more explanation. 

## Dataset
Dataset on all match statistics plus some derived features from season 2008 to 2012 is provided as an attachment supported by a data dictionary for the same.
The training dataset contains match statistics from 2008 to 2011 season. And the test dataset contains match statistics from 2012 season. 
-- train data : train.csv
-- test data  : test.csv
-- data dictionary : data_dictionary.xlsx

## Predictions
Prediction of a team (either Team 1 or Team 2) winning the match & probability score for the team winning the game. Used different types of algorithms for predicting the winner:
--RandomForestRegressor
--GradientBoostingRegressor
--XGBRegressor
--ExtraTreesRegressor
And also Grid search was used to tune the parameters.
