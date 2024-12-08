import matplotlib.pyplot as plt
import pandas as pd

# reading in dataset
df = pd.read_csv("tilsyn.csv", sep=(";"))


# filter to only retrive the Oslo establishments 
df = df[df["poststed"]== "Oslo"]

#set the total grades as different labels, add explode and precentages for better visualisations
# filter to only retrive the Oslo establishments 
zero = df.loc[df["total_karakter"]== 0].sum()[1]
one = df.loc[df["total_karakter"]== 2].sum()[1]
two= df.loc[df["total_karakter"]== 3].sum()[1]
three = df.loc[df["total_karakter"]== 3].sum()[1]
explode = (.1, .1, .1, .1)
plt.gcf().set_size_inches((10, 10))  
plt.title("Precentage of different grades", fontsize = 15)
labels = ["Grade 0" , "Grade 1", "Grade 2", "Grade 3"]
plt.pie([zero, one, two, three],explode = explode, labels = labels, autopct='%.0f%%');

plt.legend(labels, loc='upper left')
plt.show()
plt.savefig("Total_grades_pieplot.png") 