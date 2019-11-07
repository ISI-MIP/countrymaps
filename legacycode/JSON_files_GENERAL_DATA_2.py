import json
import os
import pandas
import numpy as np

text = ''

#path = 'H:/Dropbox/werk/Shared_folders_data/ISIPEDIA_IIASA/ISIPEDIA-portal/CDLX/Current_Conditions'
path = 'C:/Users/TedVeldkamp/Documents/Dropbox/werk/Shared_folders_data/ISIPEDIA_IIASA/ISIPEDIA-portal/CDLX/Current_Conditions'

input_folder = path+'/INTERMEDIATE_DATA/GENERAL/'
output_folder = path+'/INTERMEDIATE_DATA4/country_data/'

df = pandas.read_csv(input_folder+'countries_list.csv', delimiter=',')
countries = df["Abbreviation"].tolist()
 
df = pandas.read_csv(input_folder+'countries_list_full_names.csv', delimiter=',')
countries_full_names = df["Full Name"].tolist()

df = pandas.read_csv(input_folder+'indicators_list_v2.csv', delimiter=';')
ind_types = df["type"].tolist()
ind_labels = df["label"].tolist()
ind_units = df["unit"].tolist()

for index, country in enumerate(countries):
    #index = 0
    #country = 'AFG'
    
    country_name = countries_full_names[index]
    
    print(index, country)
    temp = list()
    stats = [] 
    for index2, ind in enumerate(ind_types):
        
        #index2 = 0 
        #ind = 'POP_TOTL'
        
        df = pandas.read_csv(input_folder+ind+'.csv', header=None, delimiter=',')
        #df1 = df.where((pandas.notnull(df)), None)
        #df1 = df1.values.tolist()  
        df1 = df.values.tolist()
        df1 = [x[0] for x in df1]
             
        obj = pandas.Series(df1)
        obj2 = obj.rank(method='min',ascending=False)
      
        if index2 == 0:
            obj = obj/1000000
        
        if index2 == 5:
            obj = obj/1000000
            
        df1 = obj.values.tolist()
        df2 = obj2.values.tolist()      
        no_val = 'nan'
        
        if country == 'world': stats.append({'type': ind_types[index2], 'label': ind_labels[index2], 'unit': ind_units[index2], 'value': str(np.round(df1[index],2)), 'rank': str(no_val)
        })
        if country != 'world': stats.append({'type': ind_types[index2], 'label': ind_labels[index2], 'unit': ind_units[index2], 'value': str(np.round(df1[index],2)), 'rank': str(df2[index])
        })
          
    country_type = 'country'
    if country == 'world': country_type = 'global'

    container = {}
    container['name'] = country_name
    container['type'] = country_type
    container['sub-countries'] = []
    container['stats'] = stats
    
    output_folder2 = output_folder+country+'/' 
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)
    
    with open(output_folder2+country+'_general.json', 'w') as outfile:  
      json.dump(container, outfile)  
    
 ##

 
#%%