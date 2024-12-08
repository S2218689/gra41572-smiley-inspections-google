import pandas as pd
from tabulate import tabulate

#reading in dataset
df= pd.read_csv("mat_and_google.csv")

# Group by orgnummer and name to get the rigth amount or obersvations on the Google data
df1 = df.groupby(["orgnummer","name"]).agg({"rating": "first", "price_level":"first",
             "user_ratings_total":"first"})
#print(df1)


# Using describe on dataframe to retrive statistics for the numeric google values
df = tabulate(df1.describe(),headers='keys', tablefmt='fancy_grid')
print(df)


