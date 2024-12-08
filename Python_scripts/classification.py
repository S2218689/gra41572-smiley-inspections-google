import pandas as pd
import matplotlib as plt
import numpy as np
import seaborn as sns

# imports for machine learning
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import random
from sklearn.metrics import accuracy_score
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from tabulate import tabulate
from sklearn import metrics


# reads preproccesed data ready for modelling.
df = pd.read_csv("model_data.csv", index_col = [0])
# Dummy variables for postal_code.
dummy = pd.get_dummies(df["post_code"], drop_first=True)



def get_unique_combinations(variables):
    """
    Function that finds all unique combinations of the variables.
    returns: List with lists. Each inner list contains a combination of variables.
    """
    list_combinations = []
    var_set = set(variables)
    for n in range(1,len(var_set) + 1):
        if n == 1:
            for var in variables:
                list_combinations.append([var])
        else:
            for combo in combinations(var_set, n):
                list_combinations.append(list(combo))
                
    return list_combinations


variables = ["rating", "price_level", "dine_in", "user_ratings_total", "reservable", "website", "has_summary"]
list_combinations = get_unique_combinations(variables)



def features_selection(df, postal_code = False):
    """
    Function thar tries all combinations of features on the four models KNN_C, XGB_C, RFR_C and BernoulliNB_C.
    postal_code: Boolean parameter, if set to True will use add dummy variables for postal_code on all combinations.
    Returns: List with all dictionaries with results from each model.
    can be converted into Pandas dataframe. 
    """
    res = []
    df = df.dropna()
    # Loops over all combinations of features
    for combo in list_combinations:
        X = df[combo]
        y = df["priority"]
        if postal_code:
            X = pd.concat((X,dummy),axis=1)

        # loops over 10 random seeds to test our model with differen splits. 
        for rand_int in random.sample(range(10000),1):
            scores = []

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=rand_int)
            # Loops over all models to train and test each of them. 
            for model in list([KNeighborsClassifier(28),XGBClassifier(),RandomForestClassifier(n_estimators = 100,random_state=rand_int), BernoulliNB()]):
                model.fit(X_train, y_train)
                y_hat = model.predict(X_test)
                scores.append(accuracy_score(y_test,y_hat))
            res.append({"features":combo,"KNN_Score":scores[0], "XGB_Score":scores[1], "RFC_Score": scores[2], "BNB_Score":scores[3]})

    return res




models_performance = features_selection(df)
models = pd.DataFrame(models_performance)
#print(models.sort_values("KNN_Score",ascending=False))


models["features"] = models.features.apply(str)
# Aggregate the mean for based on the combinations for the three models
mean_models = models.groupby("features", as_index =False).mean()



# Find best features for each model
RFC_score = mean_models.sort_values("RFC_Score", ascending=False).iloc[0:1,[0,3]] # CHANGE TO RFC
KNN_score = mean_models.sort_values("KNN_Score", ascending=False).iloc[0:1,[0,1]]
XGB_score = mean_models.sort_values("XGB_Score", ascending=False).iloc[0:1,[0,2]]
BNB_score = mean_models.sort_values("BNB_Score", ascending=False).iloc[0:1,[0,4]]


fin_tab = [{"Models": "Random forrest classifier", "Features":RFC_score.iloc[0,0], "Accuracy":RFC_score.iloc[0,1]},
          {"Models": "K-nearest neighbors classififer", "Features":KNN_score.iloc[0,0], "Accuracy":KNN_score.iloc[0,1]},
          {"Models": "Extreme Gradient Boosting classifier", "Features":XGB_score.iloc[0,0], "Accuracy":XGB_score.iloc[0,1]},
          {"Models": "Bernoulli Naive Bayes classifier", "Features":BNB_score.iloc[0,0], "Accuracy":BNB_score.iloc[0,1]}]

data_tab = pd.DataFrame(fin_tab).sort_values("Accuracy",ascending =False)
#data_tab
tab = tabulate(data_tab, showindex=False, headers = "keys", tablefmt="fancy_grid", floatfmt=".2f")
print(tab)



# Recreate the best model with hyperparameter tuning

features = ['website', 'has_summary', 'price_level']
X = df[features]
y = df["priority"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)



# Creating the model
from sklearn.model_selection import GridSearchCV

rfc = RandomForestClassifier() 

grid_vals = {'criterion': ['log_loss', "gini", "entropy"], 'max_features': ["log2", "sqrt", None], "random_state":[33],
            "n_estimators": [10,100,250,500,1000]}
grid_lr = GridSearchCV(estimator=rfc, param_grid=grid_vals, scoring='accuracy', 
                       cv=3, refit=True, return_train_score=True) 

grid_lr.fit(X_train, y_train)
preds = grid_lr.best_estimator_.predict(X_test)


print("ACCURACY OF THE MODEL WITH BEST PARAMETERS): ", metrics.accuracy_score(y_test, preds))



# Showing best parameters from hyperparamter tuning
rfc_frame = pd.DataFrame(grid_lr.cv_results_)
print("The best parameters fore the model are:")
print(grid_lr.best_params_)




# Creating a confusion matrix to what the real category and what the model categorized the firm as. 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20,8))
cm = confusion_matrix(y_test, preds, normalize=None)
cm_display = ConfusionMatrixDisplay(cm)
cm_display.plot(ax=ax, cmap="Blues").ax_.set_title("Random forest classifier confusion Matrix", fontsize = 15)
plt.savefig("Random_forest_C_Confusion_matrix.png")































# Code for creating a map which shows predicted classification on map.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.model_selection import train_test_split

firms = pd.read_csv("complete_dataset_matilsynet.csv")
df = df.reset_index(drop=True)
firms["latlng"] = firms.latlng.apply(lambda x: eval(x) if pd.notnull(x) else x)
df["latlng"] = df.merge(firms, on =("name","org_nr"))["latlng"]


features = ['website', 'has_summary', 'price_level', "latlng"]
X = df[features]
y = df["priority"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

X_test["predicted"] = preds

import pandas as pd
import folium
from folium.plugins import FastMarkerCluster

folium_map = folium.Map(location=[59.911491, 10.757933],zoom_start=12,tiles='cartodbpositron')

color_map = {0:"green", 1:"orange", 2:"red"}

X_test["color"] = X_test["predicted"].map(color_map)
# These two lines should create FastMarkerClusters
#cluster = FastMarkerCluster(data=firms["latlng"]).add_to(folium_map)
folium.LayerControl().add_to(folium_map)

for index, row in X_test.iterrows():
    folium.CircleMarker(location = row["latlng"],
                        radius= 2,
                        color=row["color"],
                        fill=True).add_to(folium_map)



from branca.element import Template, MacroElement


# HTML code that enables a floating legend on map
template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>jQuery UI Draggable - Default functionality</title>
  <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

  <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
  
  <script>
  $( function() {
    $( "#maplegend" ).draggable({
                    start: function (event, ui) {
                        $(this).css({
                            right: "auto",
                            top: "auto",
                            bottom: "auto"
                        });
                    }
                });
});

  </script>
</head>
<body>

 
<div id='maplegend' class='maplegend' 
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>
     
<div class='legend-title'>Predicted classification of establishments by model</div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:red;opacity:0.7;'></span>High priority</li>
    <li><span style='background:orange;opacity:0.7;'></span>Medium priority</li>
    <li><span style='background:green;opacity:0.7;'></span>Low priority</li>

  </ul>
</div>
</div>
 
</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""

macro = MacroElement()
macro._template = Template(template)

folium_map.get_root().add_child(macro)

folium_map.save("predicted_classifications_map.html")
# MAP IS SAVED AS A FILE IN YOUR FOLDER OPEN IT IN YOUR BROWSER TO SEE MAP




