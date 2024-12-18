import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from sklearn.model_selection import train_test_split

firms = pd.read_csv("complete_dataset_matilsynet.csv")
df = pd.read_csv("model_data.csv", index_col = [0]).reset_index(drop=True)
firms["latlng"] = firms.latlng.apply(lambda x: eval(x) if pd.notnull(x) else x)
df["latlng"] = df.merge(firms, on =("name","org_nr"))["latlng"]


features = ['website', 'has_summary', 'price_level', "latlng"]
X = df[features]
y = df["priority"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)




import pandas as pd
import folium
from folium.plugins import FastMarkerCluster

folium_map = folium.Map(location=[59.911491, 10.757933],zoom_start=12,tiles='cartodbpositron')

color_map = {0:"green", 1:"orange", 2:"red"}

X_test["color"] = df["priority"].map(color_map)
# These two lines should create FastMarkerClusters
#cluster = FastMarkerCluster(data=firms["latlng"]).add_to(folium_map)
folium.LayerControl().add_to(folium_map)

for index, row in X_test.iterrows():
    folium.CircleMarker(location = row["latlng"],
                        radius= 2,
                        color=row["color"],
                        fill=True).add_to(folium_map)



from branca.element import Template, MacroElement

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
     
<div class='legend-title'>True classification of establishments</div>
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

folium_map.save("True_classifications_map.html")
# MAP IS SAVED AS A FILE IN YOUR FOLDER OPEN IT IN YOUR BROWSER TO SEE MAP




