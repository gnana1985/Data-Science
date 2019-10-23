#!/usr/bin/env python
# coding: utf-8

# ## Capstone Project - The Battle of the Neighborhoods (Week 1/2)
# ## ----------------------------------------------------------------------

# ## Introduction to the Business Problem

# This project is aimed at finding an optimal neighborhood location for a restaurant. **Specifically, this report is to advice a Thai restaurant chain that is interested in opening an outlet in Toronto.**
# 
# In Canada, especially in Toronto, being a food capital of Canada, Thai food is becoming very popular.Thai food has a multi ethnic appeal. It is basically enjoyed by South East Asian, Chinese and Indian community which is a sizeable population in Toronto.The chefs of this chain can modify the basic curry local spices and understand the need for western tastes and modified the level of spices and hotness. This chain has several outlets in the USA running successfully and wants to start operations in Canada.
# In Toronto, there are several successful Thai restaurants already and the question to be answered is with regard to the choice of location to beat existing competition and yet be profitable. That is to be able to find new catchment areas with similar characteristics as that of locations corresponding to the existing competitors.
# 
# We will use our data science techniques to generate a few most promising neighborhoods based on this criteria. 

# ## Data Section

# Key data sets required to proceed with solution are:
# 
# 1. Neighborhood details for Toronto. Web crawling to understand the Toronto neighbourhood preferences, demographics and its food industry. https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M, will be scraped using the BeautifulSoup packageto get the basic neighborhood data (Canada Boroughs,Neighbourhood and Postal code) . This will be augmented by geographical coordinates (Lattitude and Longitude) using the data in the csv file : http://cocl.us/Geospatial_data
# 
# 2. Venue details that includes Thai and non Thai restaurants. For this Foursqaure API Venues data will be used to pull 100 venues in each neighbourhood within a radius of 500m that will include the venue name, venue category, longitude and lattitude details.
# 
# Data analysis:
# 1. Basic clean up of the datasets will be performed to make it suitable for running any data science analytics
# 2. Neighborhoods will be clustered and  analyzed to understand the spending patterns and restaurant density in general. Geographical proximity and its association with the demographics of the physical locations will be analyzed which will provide key pointers while recommending the location for Thai restaurants
# 3. Density and geospatial spread of the existing Thai restaurants will be analyzed on the map and superimposed on the neighborhood clusters.
# 

# In[50]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

get_ipython().system('conda install -c anaconda beautifulsoup4 --yes')

print('Libraries imported.')


# ## 1. Download and Explore Dataset

# In[51]:


#### Scraping wikipedia data to get the list of Canada neighbourhoods


# In[52]:


from bs4 import BeautifulSoup


# In[53]:


html = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"

## use bs4 scraping content from url
response = requests.get(html)

html_soup = BeautifulSoup(response.text, 'html.parser')
type(html_soup)


# In[54]:


pip install lxml


# In[55]:


# get tables
table = html_soup.find_all('table')[0] 
df = pd.read_html(str(table))[0]

# get a record from table
df.head()


# ### 1. Cleaning the neighbourhood data

# In[56]:


## drop column which Borough is not assigned
df_bor = df[~df.Borough.isin(['Not assigned'])]

## combine same Postcode into one column
df_combine = df_bor.groupby(['Postcode','Borough'])['Neighbourhood'].apply(','.join).reset_index()

## assign same value of Borough to Neighbourhood which is not assigned
df_combine.loc[df_combine['Neighbourhood']=="Not assigned",'Neighbourhood'] = df_combine.loc[df_combine['Neighbourhood']=="Not assigned",'Borough']
df_combine.head()


# In[57]:


df_combine.shape


# ### Assigning Latitude and Longitude to the neighbourhood data

# In[58]:


a = pd.read_csv('https://cocl.us/Geospatial_data')
gdf = pd.DataFrame(a)


# In[59]:


## assign lat,long to postcode dataframe
df_post = pd.merge(df_combine,gdf,how='left',left_on='Postcode',right_on='Postal Code')
df_post.drop('Postal Code',axis=1,inplace=True)
df_post.head()


# In[60]:


df_post.shape


# In[61]:


address = 'Toronto, ON'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto City are {}, {}.'.format(latitude, longitude))


# In[62]:


# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(df_post['Latitude'], df_post['Longitude'], df_post['Borough'], df_post['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[63]:


#We are going to work with only the boroughs that contain the word "Toronto".

df_t = df_post[df_post['Borough'].str.contains("Toronto")].reset_index(drop=True)
df_t.head()


# #### Define Foursquare Credentials and Version

# In[64]:


CLIENT_ID = 'YSH1ABHBNWGMJW4T5LP2ETHXS40D5TUBWJ1JFE4MLET2SITO' # your Foursquare ID
CLIENT_SECRET = 'GHCI4N3ONG5XOMAG3ETKE4DCO0KWN0YHAJOPI1XI1BM2BJAN' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# #### Let's explore the first neighborhood in our dataframe

# In[65]:


df_t.loc[0, 'Neighbourhood']


# #### Get the neighborhood's latitude and longitude values.

# In[66]:


neighborhood_latitude = df_t.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = df_t.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = df_t.loc[0, 'Neighbourhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# #### Now, let's get the top 100 venues that are in The Beaches within a radius of 500 meters.

# In[67]:


LIMIT = 100 # limit of number of venues returned by Foursquare API

radius = 500 # define radius

# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL




# In[68]:


results = requests.get(url).json()


# In[69]:


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
    
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()   


# In[70]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# ## 2. Exploring Neighborhoods in Toronto

# In[71]:


#function to repeat the same process to all the neighborhoods in Toronto
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[72]:


# Using the above function on each neighborhood to create a new dataframe called toronto_venues
toronto_venues = getNearbyVenues(names=df_t['Neighbourhood'],
                                   latitudes=df_t['Latitude'],
                                   longitudes=df_t['Longitude']
                                  )
print(toronto_venues.shape)
toronto_venues.head()


# In[73]:


#Let's check the size of the resulting dataframe
print(toronto_venues.shape)
toronto_venues.head()


# In[74]:


#Let's check how many venues were returned for each neighborhood

toronto_venues.groupby('Neighborhood').count()


# In[75]:


#Unique venue categories in Toronto

print('There are {} uniques categories.'.format(len(toronto_venues['Venue Category'].unique())))


# ## 3. Analyze Each Neighborhood

# In[76]:


# one hot encoding
toronto_onehot = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
toronto_onehot['Neighbourhood'] = toronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# In[77]:


toronto_onehot.shape


# In[78]:


#Grouping rows by neighborhood and by taking the mean of the frequency of occurrence of each category
toronto_grouped = toronto_onehot.groupby('Neighbourhood').mean().reset_index()


# In[79]:


toronto_grouped.shape


# In[80]:


#function to sort the venues in descending order
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[81]:


#Dataframe for the top 10 venues for each neighborhood.

num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = toronto_grouped['Neighbourhood']

for ind in np.arange(toronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(toronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ## 4. Cluster Neighborhoods

# In[82]:


#Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.


# In[83]:


kclusters = 5

toronto_grouped_clustering = toronto_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[84]:


#Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# In[85]:


neighborhoods_venues_sorted.head()


# In[86]:


toronto_merged = df_t

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on='Neighbourhood')

toronto_merged.head() # check the last columns!


# In[87]:


# Create map to visualize the clusters
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ## 5. Examine Clusters

# #### Cluster0

# In[88]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 0, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]


# ##### Cluster0 seems to have venues that are popular in the busy areas as is probably marks the business district of Toronto with lots of cafes, pubs and light weight entertainment centers to suit the busy lifestyle of the working class.
# This cluster also encompasses the bulk of Toronto neighbourhoods indicating the lifestyle or key characteristic of the city itself.

# #### Cluster1

# In[89]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 1, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]


# ##### Cluster1 seems like a residential zone with the basic and essential venues particularly useful for the vulnerable ie children and aged.¶
# There is also geographical proximity of these neighourhoods most stretching away from downtown to the north

# #### Cluster2

# In[90]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 2, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]


# ##### Cluster2 seems to have venues characteristic of people who want to enjoy peaceful and leisure living.
# Is a residential locality surrounded by posh neighbourhoods. Is geographically toward the north, away from the key business district.

# #### Cluster3

# In[91]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 3, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]


# ##### Cluster3 seems like a zone inhabited with health conscious poeple.
# Wiki notes on Moore park - "Moore Park is one of Toronto's most affluent neighbourhoods."

# #### Cluster4

# In[92]:


toronto_merged.loc[toronto_merged['Cluster Labels'] == 4, toronto_merged.columns[[2] + list(range(5, toronto_merged.shape[1]))]]


# ##### Cluster4 seems to have venues particularly suited for a peaceful community and large greenspace in central Toronto and seems prosperous.
# It is aligned to wikipedia content on Rosedale: "It is located north of Downtown Toronto and is one of its oldest suburbs. It is also one of the wealthiest and most highly priced neighbourhoods in Canada.[2] Rosedale has been ranked the best neighbourhood in Toronto to live in by Toronto Life.[3] It is known as the area where the city's 'old money' lives,[4] and is home to some of Canada's richest and most famous citizens including Gerry Schwartz, founder of Onex Corporation, and Ken Thomson of Thomson Corporation, the latter of whom was the richest man in Canada at the time of his death in 2006"

# ### Finding top 10 common restaurant categories in Toronto

# In[93]:


t_venue_cat_common=toronto_venues.groupby(['Venue Category']).size().reset_index(name='count').sort_values('count', ascending=False)
t_res=t_venue_cat_common[t_venue_cat_common['Venue Category'].str.contains("Restaurant")].reset_index(drop=True)
t_res.head(10).plot('Venue Category', 'count', kind='bar', figsize=(8,4), width=.25,colormap='Paired')


# ### Finding neighborhoods with high restaurant density in Toronto

# In[94]:


toronto_res=toronto_venues[toronto_venues['Venue Category'].str.contains("Restaurant")].reset_index(drop=True)
#toronto_thai
toronto_res_nei=toronto_res.groupby(['Neighborhood']).size().reset_index(name='count').sort_values('count', ascending=False)
toronto_res_nei


# ### Finding neighborhoods with Thai restaurants in Toronto

# In[95]:


toronto_thai=toronto_venues[toronto_venues['Venue Category'].str.contains("Thai Restaurant")].reset_index(drop=True)
#toronto_thai
toronto_thai_nei=toronto_thai.groupby(['Neighborhood']).size().reset_index(name='count').sort_values('count', ascending=False)
toronto_thai_nei


# In[96]:


toronto_thai_nei.plot('Neighborhood', 'count', kind='bar', figsize=(8,4), width=.25,colormap='Paired')


# In[97]:


# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
toronto_thai_merged = toronto_merged.join(toronto_thai_nei.set_index('Neighborhood'), on='Neighbourhood').sort_values('count', ascending=False)
toronto_thai_merged.dropna(axis=0, subset=('count', ), inplace=True)
toronto_thai_merged


# It is clear that Cluster3 which houses all the existing Thai restaurants and is probably better suited for a new one as well.
# Let us find out which is the best neighbourhood in Cluster3 that is suited for the new restaurant.
# 
# 
# 

# In[98]:


# Create map to superimpose Italian restaurants on the cluster map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(toronto_merged['Latitude'], toronto_merged['Longitude'], toronto_merged['Neighbourhood'], toronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
    
# add the Thai restaurants as yellow circle markers
for lat, lng, label in zip(toronto_thai['Neighborhood Latitude'], toronto_thai['Neighborhood Longitude'], toronto_thai['Venue']):
    folium.features.CircleMarker(
        [lat, lng],
        radius=2,
        color='black',
        popup=label,
        fill = True,
        fill_color='yellow',
        fill_opacity=0.6
    ).add_to(map_clusters)
    
map_clusters


# On the basis of the visual inspection of the existing thai restaurants in Cluster0, 
# any new restaurant can be positioned in the western Cluster0 neighborhoods ie in the area between High Park and Salad King which shares the characteristics of neighborhoods in which the restaurant business is thriving.
# Top picks are the neighborhoods of Dovercourt Village and Christie given the low restaurant density in that, competition is less in that area and hence can prove to be quite profitable.
# 

# In[ ]:





# In[ ]:




