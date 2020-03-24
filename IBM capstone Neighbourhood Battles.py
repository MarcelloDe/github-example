#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests # library to handle requests


# In[2]:


import pandas as pd # library for data analsysis


# In[3]:


import numpy as np # library to handle data in a vectorized manner


# In[4]:


import random # library for random number generation


# In[5]:


from geopy.geocoders import Nominatim 


# In[6]:


# libraries for displaying images
from IPython.display import Image 
from IPython.core.display import HTML 


# In[7]:


# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

import folium # plotting library

print('Folium installed')
print('Libraries imported.')


# In[8]:


CLIENT_ID = 'JHZW5GAKFK2XE44XRTB0TYSWJJ4VYRFN0IA0LN5H2JD04I3Y' # your Foursquare ID
CLIENT_SECRET = 'E5PCPKDBNTUYD305KFQUPUXDLKGCJMRZQ53Y41UVJBTUFKCQ' # your Foursquare Secret
VERSION = '20180604'
LIMIT = 2000
print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[9]:


df = pd.read_csv('C:/Users/Marcello/Desktop/Coursera Data Science/9 Applied Data Science Capstone/Week Four- The Battle of Neighborhoods/Educational_Institutions.csv')


# In[10]:


df


# In[11]:


address = 'Hamilton, Ontario'

geolocator = Nominatim(user_agent="foursquare_agent")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print(latitude, longitude)


# In[12]:


search_query = 'restaurant'
radius = 10000
print(search_query + ' .... OK!')


# In[13]:


url = 'https://api.foursquare.com/v2/venues/search?client_id={}&client_secret={}&ll={},{}&v={}&query={}&radius={}&limit={}'.format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, search_query, radius, LIMIT)
url


# In[14]:


results = requests.get(url).json()
results


# In[15]:


# assign relevant part of JSON to venues
venues = results['response']['venues']

# tranform venues into a dataframe
dataframe = json_normalize(venues)
dataframe.head()


# In[16]:


# keep only columns that include venue name, and anything that is associated with location
filtered_columns = ['name', 'categories'] + [col for col in dataframe.columns if col.startswith('location.')] + ['id']
dataframe_filtered = dataframe.loc[:, filtered_columns]

# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# filter the category for each row
dataframe_filtered['categories'] = dataframe_filtered.apply(get_category_type, axis=1)

# clean column names by keeping only last term
dataframe_filtered.columns = [column.split('.')[-1] for column in dataframe_filtered.columns]

dataframe_filtered


# In[17]:


food = pd.DataFrame(dataframe_filtered)


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


food['categories'].value_counts()


# In[20]:


print(food.shape)
food.head()


# In[21]:


#GROUPBY postcode and hoods
food2 = food.groupby(['lat','lng'])['name'].apply(lambda x: ", ".join(x.astype(str))).reset_index()
food2 = food.sample(frac=1).reset_index(drop=True)


# In[22]:


venues_map = folium.Map(location=[latitude, longitude], zoom_start=13) 

# add a red circle marker to represent post secondary
folium.CircleMarker(
    [latitude, longitude],
    radius=1,
    color='white',
    popup='Hamilton, Ontario',
    fill = True,
    fill_color = 'white',
    fill_opacity = 0
).add_to(venues_map)

# blue marks coffee shop locations
for lat, lng, label in zip(food2['lat'],food2['lng'], food2['name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        color='blue',
        popup= label,
        fill = True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(venues_map)
 

for lat, lng, label in zip(df['LATITUDE'], df['LONGITUDE'], df['Name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        color='yellow',
        popup=label,
        fill = True,
        fill_color='yellow',
        fill_opacity=0.6
    ).add_to(venues_map)

# display map


# In[23]:


food2.groupby('categories').count()


# In[24]:


print('There are {} uniques categories.'.format(len(food['categories'].unique())))


# In[25]:


food_onehot = pd.get_dummies(food2[['categories']], prefix="", prefix_sep="")


# In[26]:


food_onehot['categories'] = food2['categories'] 


# In[27]:


food_grouped = food_onehot.groupby('categories').mean().reset_index()
food_grouped


# In[28]:


food_grouped["categories"]= food_grouped["categories"].str.replace('Vietnamese Restaurant', 'Vietnamese', case = False) 


# In[29]:


from sklearn.cluster import KMeans


# In[30]:


#CLUSTERING
# set number of clusters
kclusters = 2

food_grouped_clustering = food_grouped.drop('categories', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(food_grouped_clustering)


# In[31]:


# add clustering labels
food_grouped.insert(0, 'Cluster Labels', kmeans.labels_)


# In[32]:


# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
food3 = food2.join(food_grouped.set_index('categories'), on='categories')

food3.head() # check the last columns!


# In[33]:



# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[34]:


# create map
map_cluster = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(food2['lat'], food2['lng'], food3['categories'], food3['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow,
        fill=True,
        fill_color=rainbow,
        fill_opacity=0.7).add_to(venues_map)
       
venues_map


# In[ ]:





# In[ ]:





# In[ ]:




