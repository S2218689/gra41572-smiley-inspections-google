import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as rfr
import random
from tabulate import tabulate

# Read the csv-file
model_data = pd.read_csv("model_data.csv",dtype={"post_code": object}, index_col=[0])

# Form list of possible combinations
variables = ["rating", "price_level", "dine_in", "user_ratings_total", "reservable", "website", "has_summary"]
list_combinations = []
var_set = set(variables)
for n in range(1,len(var_set) + 1):
    if n == 1:
        for var in variables:
            list_combinations.append([var])
    else:
        for combo in combinations(var_set, n):
            list_combinations.append(list(combo))

# Runs a linear regression, eXtreme Gradient Bosting Regression, and Randon Forest Regression
# for all combinations with 10 different random states
# Returns the results from the regressions in a list
def run_models(with_post_code=False):
    res = []
    for combo in list_combinations:
        x = model_data[combo]
        y = model_data["weighted_score"]
        if with_post_code:
            dummy = pd.get_dummies(model_data["post_code"], drop_first=True)
            x = pd.concat((x,dummy),axis=1)

        for rand_int in random.sample(range(10000),10):
            r2_scores = []

            data_train, data_test, target_train, target_test = train_test_split(x, y, test_size=0.3, random_state=rand_int)

            for model in list([lm.LinearRegression(),xgb.XGBRegressor(),rfr()]):
            
                model.fit(data_train, target_train)
                target_pred = model.predict(data_test)
                r2_scores.append(r2_score(target_test,target_pred)*100)

            res.append({"features":combo[:-1],"LM_R2":r2_scores[0], "XGB_R2":r2_scores[1], "RFR_R2": r2_scores[2]})
    return res

# Make the results into a dataframe and apply features as string
model_score = pd.DataFrame(run_models())
model_score["features"] = model_score["features"].apply(str)

# Aggregate the mean for based on the combinations for the three models
mean_models = model_score.groupby("features", as_index=False).agg("mean")

# Sort on Random Forest Regression 
mean_models = mean_models.sort_values("RFR_R2", ascending=False)

print(tabulate(mean_models.head(), showindex=False, headers=["Features", "LM R2 mean (%)", "XGB R2 mean (%)", "RFR R2 mean (%)"], tablefmt="fancy_grid", floatfmt=".2f"))
