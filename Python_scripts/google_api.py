import pandas as pd
import requests

# Read csv-file from NFSA
df = pd.read_csv("tilsyn.csv", encoding="utf-8", sep =";")

# Turn into dataframe and filter on Oslo 
ot = df[df["poststed"] == "OSLO"].copy().reset_index(drop=True)

# Temporary dataframe to drop duplicates and reset index
mid_ot = ot.copy().drop_duplicates(subset=["navn", "orgnummer"])
mid_ot.reset_index(drop=True, inplace=True)

# API-requests
# One to find the id and one to get details
# Returns the google details, if any, and the organization number and name (for merging purposes)
API_KEY = "Insert API key here"

def getInfo(name, address, orgnum):
    payload={}
    headers = {}
    try:
        url_id = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={name+' '+address}&inputtype=textquery&key={API_KEY}"
        response = requests.request("GET", url_id, headers=headers, data=payload)
        id = response.json()["candidates"][0]["place_id"]
        url_info = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={id}&fields=name%2Crating%2Cbusiness_status%2Cdine_in%2Ceditorial_summary%2Cprice_level%2Creservable%2cuser_ratings_total%2ctypes%2cwebsite&key={API_KEY}"
        response = requests.request("GET", url_info, headers=headers, data=payload)
        res = response.json()["result"]
        res["orgnummer"] = orgnum
        res["navn"] = name
    except Exception as error:
        print(error)
        res = {"orgnummer": orgnum, "navn":name}
    return res

# Loop through the firms from NFSA and add results to a list
google_info = []
for i, row in mid_ot.iterrows():
    google_info.append(getInfo(row.navn, row.adrlinje1, row.orgnummer))

# Make list a dataframe
gdf = pd.DataFrame(google_info)

# Merge on organization number and name
total = ot.merge(gdf, on = ["orgnummer", "navn"])

# Save as csv
total.to_csv("mat_and_google.csv")