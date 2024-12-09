import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
This code snippet can be run to remake the plot that shows labels given
to each firm based on weighted_score.
"""

#Import dataset
df = pd.read_csv("model_data.csv",dtype={"post_code": object})


# Uses seaborn and matplotlib to genereate jointplot which shows both
# distribution and how each category is defined. 
ax = sns.jointplot(data=df, x = "weighted_score", y="priority")
ax.set_axis_labels("Weighted Score","priority", fontsize=12)
ax.fig.suptitle("Categorisation results")
ax.fig.subplots_adjust(top=0.95)
plt.show()
