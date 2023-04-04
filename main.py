# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:29:21 2021

@author: seanb126
"""

#imported libraries
import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import PolyCollection
import json
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import tree

#for console colours
class tcolours:
    HINT = '\033[33m'
    WELCOME = '\033[32m'
    WARNING = '\033[31m'
    DEF = '\033[39m'
    
#program start
print(f"{tcolours.WELCOME}\nApplied Data Mining Project")
print('Based on Transcripts from 1856-1914, for ships registered at the port of Aberystwyth, Ceredigion')
print(f'For the Sylfaen Treftadaeth /Local Heritage Foundation{tcolours.DEF}\n')
print(f'{tcolours.WARNING}\nWarning... this can take a few seconds{tcolours.DEF}\n')

#stops unecessary warnings popping up
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

#database connection
#must be done on shellsrv
user = ''
dbpath = '' #removed for security
password = 'available upon request' # i never changed the password
                                    # its the same as on the email i originally received 
connection_string = 'mongodb://'+user+':'+password+'@'+dbpath

client = MongoClient(connection_string)

# accesses the db collection
# db = client. #.username
col = db
col = db[''] #username removed




# finds data within mongodb collection
cursor = col.find()

# dataframe of mongodb data
df = pd.DataFrame(list(cursor))


# expands the mariner data into new dataframe
dfe = df.assign(var1=df['mariners'].str.split(',')).explode('mariners')
dfe = pd.DataFrame(dfe['mariners'].values.tolist())

#for ease chages column name
dfe = dfe.rename(columns={'this_ship_capacity': 'rank'})
#cleans data
dfe['name'] = dfe['name'].str.replace('[^a-zA-Z]', ' ', regex = True)
dfe['name'] = dfe['name'].replace('\s+', ' ', regex=True)
dfe['name'] = dfe['name'].str.strip().str.replace(" ", "_", regex=True)
dfe['name'] = dfe['name'].str.lower()
# clean place of birth
# issue with adding it to previous clean
dfe['place_of_birth'] = dfe['place_of_birth'].str.replace('[^a-zA-Z]', ' ', regex=True)
dfe['place_of_birth'] = dfe['place_of_birth'].replace('\s+', ' ', regex=True)
dfe['place_of_birth'] = dfe['place_of_birth'].str.strip().str.replace(" ", "_", regex=True)
dfe['place_of_birth'] = dfe['place_of_birth'].str.lower()

#for part two of project, clean dfe
dfe['last_ship_name'] = dfe['last_ship_name'].str.replace('[^a-zA-Z]', ' ', regex=True)
dfe['last_ship_name'] = dfe['last_ship_name'].replace('\s+', ' ', regex=True)
dfe['last_ship_name'] = dfe['last_ship_name'].str.strip().str.replace(" ", "_", regex=True)
dfe['last_ship_name'] = dfe['last_ship_name'].str.lower()

#dfe converts joining port to datetime
dfe['this_ship_joining_date'] =  pd.to_datetime(dfe['this_ship_joining_date'], format='%Y-%m-%d', errors='coerce')


# sorts data by name, age, place of birth and year of birth
dfe = dfe.sort_values(by=['name','age', 'place_of_birth', 'year_of_birth'], ignore_index=False)
dfe = dfe.reset_index()
del dfe['index']


#dataframe for name search 
df3 = dfe
nan_value = float("NaN") #Convert NaN values to empty string.
df3.replace("", nan_value, inplace=True)
df3.dropna(subset = ["name", 'place_of_birth'], inplace=True)
df3 = df3.reset_index()
del df3['index']
#df3 = df3.drop_duplicates(subset=['place_of_birth'], keep='first')
df3['sailor'] = df3['name'] +' '+ df3['place_of_birth']
df3 = df3.drop_duplicates(subset=['sailor'], keep='first')

# saves output to .csv files
dfe.to_csv('output.csv') # filtered dataframe
df3.to_csv('output_short.csv') # search criteria df

# organise/identify sailors
#print(f"{tcolours.WELCOME}\nApplied Data Mining Project")
#print('Based on Transcripts from 1856-1914, for ships registered at the port of Aberystwyth, Ceredigion')
#print(f'For the Sylfaen Treftadaeth /Local Heritage Foundation{tcolours.DEF}\n')

# functions for Part 1

def name_search(): # allows user to search sailor names
    def close_matches(string):
        global df3
        get_sr2 = df3['sailor'][df3['sailor'].str.startswith(string[:2])]
        if len(get_sr2) !=0:
            return get_sr2.tolist()
        else:
            return ''
    while True:
        print('ENTER SAILOR NAME AND PLACE OF BIRTH')
        print('Suggestions will be made if no direct matches are found')
        get_id = input(': ')
        get_sr = df3['sailor'][df3['sailor'].isin([get_id])]
        if len(get_sr) != 0:
            #print(get_sr.iloc[0])
            criteria=(get_sr.iloc[0])
            search_results(criteria)
            break
        elif close_matches(get_id):
            print("Do you mean one of the following?")
            print(close_matches(get_id),'\n')
            continue
        else:
            print("No match found. Please try again.")
            continue


def search_results(criteria): # presents search results
    dfe_search = dfe
    dfe_search['sailor'] = dfe_search['name'] +' '+ dfe_search['place_of_birth']
    cri_results = dfe_search.loc[dfe_search['sailor'] == criteria]
    
    print('\n\nFor full results type "full"')
    print('For reduced results type "reduced"')
    print('To search again type "search"')
    print('To return to the main menu type "menu"')
    r = (input(': '))
    if r in ['full', 'FULL', 'Full', 'F', 'f']:
      search_results(criteria)
    elif r in ['reduced', 'REDUCED', 'Reduced', 'R', 'r']:
      print(cri_results[['name', 'age', 'place_of_birth', 'rank']])
      search_results(criteria)
    elif r in ['menu', 'MENU', 'Menu', 'M', 'm']:
      command()
    if r in ['search','SEARCH', 'Search', 's', 'S']:
      name_search()
    elif r in ['exit', 'EXIT','Exit','e', 'E']:
      print('Program closing down...')
      # waits three seconds before closing the program down
      time.sleep(1)
    else:
      print('HINT: remember to keep your response in lower case :)')
      search_results(criteria)

def command(): # centre for the programs functionality
    print('Welcome: You can use the following commands to navigate and use this program')
    print('"search", "visuals", "help", "predictions", and "exit"')
    print(f"{tcolours.HINT}\nHINT! you can just type the first letter instead of the full command{tcolours.DEF}")
    usr_in = (input('What would you like to do? '))
    if usr_in in ['search','SEARCH', 'Search', 's', 'S']:
        #print(df3)
        print('\nWARNING! please input in the following format : firstname_lastname place_of_birth')
        name_search()
    elif usr_in in ['visuals', 'Visuals', 'VISUALS','v', 'V']:
        visuals()
    elif usr_in in ['predictions', 'Predictions', 'PREDICTIONS','p', 'P', 'predict']:
        ship_menu()
    elif  usr_in in['help', 'HELP', 'Help', 'h']:
        print(f'{tcolours.HINT}\n"help": shows all commands available to the user')
        print('"search": allows the user to search for a sailor')
        print('"visuals": will take you to the visualisations menu')
        print('"predictions": will take you to the ship predicitons menu')
        print(f'"exit": will close the program and is available on all menus{tcolours.DEF}\n')
        command()
    elif usr_in in ['exit', 'EXIT','Exit','e', 'E']:
      print('Program closing down...')
      # waits three seconds before closing the program down
      time.sleep(1)
      
    else:
      print('Type "help" to see all the commands')
      command()

def visuals():
    print('For the visualisation of all ranks in data set type "ranks"')
    print('For the promotion timelines type "promo"')
    print('To return to the main menu type "menu"')
    usr_in = (input(': '))
    if usr_in in ['ranks', 'RANKS', 'Ranks', 'r', 'R']:
        rank_vis()
        visuals()
    elif usr_in in ['promo', 'PROMO', 'Promo', 'P', 'p']:
        promo_vis()
        visuals()
    elif usr_in in ['menu', 'MENU', 'Menu', 'M', 'm']:
        command()
    elif usr_in in ['exit', 'EXIT','Exit','e', 'E']:
        print('Program closing down...')
      # waits three seconds before closing the program down
        time.sleep(1)
    else:
        print('remember you can just type the first letter')
        visuals()
        
        

def rank_vis():
#test values
# below organise and clean the rank data
    rank_count = dfe['rank']
    rank_count = rank_count.str.replace('[^a-zA-Z]', ' ', regex = True)
    rank_count = rank_count.str.replace('blk', ' ', regex = True)
    rank_count = rank_count.replace('\s+', ' ', regex=True)
    rank_count = rank_count.str.strip().str.replace(" ", "_", regex=True)
    rank_count = rank_count.str.lower()
    rank_count = rank_count.str.replace(r'\b(\w{1,2})\b', '')
    nan_value = float("NaN") 
    rank_count.replace("", nan_value, inplace=True)
    rank_count.fillna(value='other / unrecorded', inplace = True)
    rank_count.dropna(inplace=True)
    rank_count = rank_count.value_counts().loc[lambda x : x>5]
    
#visualisation
    rank_freq = rank_count.plot(kind='bar',figsize=(14,8),title = 'Frequency of Sailor Ranks')
    rank_freq.set_xlabel('Rank Titles')
    rank_freq.set_ylabel('Frequency')
    plt.show()
    #print('The figure will be automatically saved as "rank_freq.png"')
    #plt.savefig('rank_freq.png')
    # promo_vis()
    #print('\n Returning you to the start...')


#reusable sailor function
def sailor_find(name):
    get_id = '{}'.format(name)
    get_sr = df3['sailor'][df3['sailor'].isin([get_id])]
    if len(get_sr) != 0:
      criteria=(get_sr.iloc[0])
    dfe_search = dfe
    dfe_search['sailor'] = dfe_search['name'] +' '+ dfe_search['place_of_birth']
    cri_results = dfe_search.loc[dfe_search['sailor'] == criteria]
    return cri_results
    #print(cri_results)

#this function will show the promotion ranks of two sailors
def promo_vis():
    #finds the data on francis evans
    print('\nHere you can choose between two sailors')
    print('and compare the timelines of their promotions')
    print('HINT! A good comparison which was used for testing were:')
    print('"francis_evans new_quay" and "thomas_jones aberystwyth"')
    get_s1 = (input('Sailor 1: '))
    #sailor_find(get_id)
    j = sailor_find(get_s1)
    #thomas jones data
    get_s2 = (input('Sailor 2: '))
    t = sailor_find(get_s2)
    
    j['rank_id'] = j.groupby('rank').ngroup()
    j.this_ship_joining_date.dropna(inplace=True)
    j.sort_values(by='this_ship_joining_date', inplace=True, ignore_index=False)
    j = j.reset_index()
    
    #prints thomas jones' data
    t['rank_id'] = t.groupby('rank').ngroup()
    t.this_ship_joining_date.dropna(inplace=True)
    t.sort_values(by='this_ship_joining_date', inplace=True, ignore_index=False)
    t = t.reset_index()
    
    t.to_numpy()
    x1 = t['this_ship_joining_date']
    y1 = t['rank']

    j.to_numpy()
    x2 = j['this_ship_joining_date']
    y2 = j['rank']
    
    fig, axs = plt.subplots(3)
    axs[0].plot(x1, y1, marker='o', color='b')
    axs[1].plot(x2, y2, marker='o', color='r')
    axs[2].plot(x1, y1, marker='o', color='b')
    axs[2].plot(x2, y2, marker='o', color='r')

#sets titles
    fig.suptitle('Promotion Timelines')
    axs[0].title.set_text('{}'.format(get_s2))
    axs[1].title.set_text('{}'.format(get_s1))
    axs[2].title.set_text('{} + {}'.format(get_s1, get_s2))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
def ship_menu():
    sailor_count = []   
    print('\nTo see the sailor count histogram and ship path predictions, type "start"')
    #print('For the destinations and predictions of a ships path, type "path"')
  #print('For the predicitions of a ships path, type "predict"')
    print('To return to the main menu type "menu"')
    usr_in = (input(': '))
    if usr_in in ['start', 'START', 'Start','s', 'S']:
      #sailor names
        sailor_count = df[['vessel name', 'mariners']]
    #below is a quick fix for the misspelling of Aberystwyth
        sailor_count['vessel name'] = sailor_count['vessel name'].replace({'Aberystywyth': 'Aberystwyth'}, regex=True)
        sailor = sailor_count.assign(var1=sailor_count['mariners'].str.split(',')).explode('mariners')
        sailor = pd.DataFrame(sailor['mariners'].values.tolist())
    
    #sailor = sailor[['name', 'place_of_birth', 'age']]

    
    
        sailor_count = sailor_count.assign(var1=sailor_count['mariners'].str.split(',')).explode('mariners')
        sailor_count['mariners'] = pd.DataFrame(sailor_count['mariners'].values.tolist())

        sailor_count["id"] = sailor.index
        sailor['id'] = sailor.index
        sailor_count = pd.merge(sailor_count, sailor, on='id', how='outer')
    
     #cleans data
        sailor_count['name'] = sailor_count['name'].str.replace('[^a-zA-Z]', ' ', regex = True)
        sailor_count['name'] = sailor_count['name'].replace('\s+', ' ', regex=True)
        sailor_count['name'] = sailor_count['name'].str.strip().str.replace(" ", "_", regex=True)
        sailor_count['name'] = sailor_count['name'].str.lower()
      

    #cleans data
        nan_value = float("NaN") #Convert NaN values to empty string.
        sailor_count.replace("", nan_value, inplace=True)
        sailor_count['sailor'] = sailor_count['name'] +' '+ sailor_count['place_of_birth']
        sailor_count = sailor_count.drop_duplicates(subset=['sailor'], keep='first')
    
    #print and export
        sailor_count.to_csv('sailors.csv')
        fig = (sailor_count['vessel name'].value_counts().plot(kind='barh', color=['b','r']))
        fig.title.set_text('Number of Crew on Each Ship')
        plt.tight_layout()
        plt.show()
      
        df5 = sailor_count[['vessel name', 'this_ship_joining_port','this_ship_leaving_port']]
        df5['root'] = 'aberystwyth'
      #cleaning
        df5['this_ship_joining_port'] = df5['this_ship_joining_port'].str.replace('[^a-zA-Z]', ' ', regex = True)
        df5['this_ship_joining_port'] = df5['this_ship_joining_port'].replace('\s+', ' ', regex=True)
        df5['this_ship_joining_port'] = df5['this_ship_joining_port'].str.strip().str.replace(" ", "_", regex=True)
        df5['this_ship_joining_port'] = df5['this_ship_joining_port'].str.lower()
        df5.reset_index(drop=True, inplace=True)
    
    #leaving
        df5['this_ship_leaving_port'] = df5['this_ship_leaving_port'].str.replace('[^a-zA-Z]', ' ', regex = True)
        df5['this_ship_leaving_port'] = df5['this_ship_leaving_port'].replace('\s+', ' ', regex=True)
        df5['this_ship_leaving_port'] = df5['this_ship_leaving_port'].str.strip().str.replace(" ", "_", regex=True)
        df5['this_ship_leaving_port'] = df5['this_ship_leaving_port'].str.lower()
        vc = df5['this_ship_leaving_port'].value_counts()
        df5.this_ship_leaving_port.loc[df5['this_ship_leaving_port'].isin(vc.index[vc > 1])]
    
        df5.replace("blk", nan_value, inplace=True)
        df5.dropna(inplace=True)
    
        print('\nPlease type the name of the ship you would like to view')
        print(F'\n{tcolours.HINT}HINT: "Acorn" was used for testing and produces valuable results{tcolours.DEF}')
        print(pd.Series(df['vessel name']))
    #print('HINT: or type "list" to list all ship options')
        shipname = (input(': '))
        df6= df5[df5['vessel name'].str.contains('{}'.format(shipname))]
    
    #creates sixth dataframe
    # renames and reorganises df to match task

        df6 = df6.rename(columns={'this_ship_joining_port': 'b1'})
        df6 = df6.rename(columns={'this_ship_leaving_port': 'b2'})
        df6 = df6[['vessel name', 'root', 'b1', 'b2']]
        dfr = df6
    
    #only includes most common
    
    
        df7 = pd.DataFrame()
        df6.reset_index()
        df6['b1']
        df6['path'] = df6.index
    #find 10 most popular ports for branch 1
        branch1 = df6.b1.value_counts().nlargest(10).index
    #mask boolean to determine popularity
        df6['b1_mask'] = df6.b1.isin(branch1.tolist())
        df6 = df6[df6['b1_mask'] == True]
        branch2 = df6.groupby(['b1'])
    
    
    
    
        fig, axs = plt.subplots()
        plt.tight_layout()
        axs.title.set_text('Common Paths of Ships from Aberystwyth')
        axs.set_xlabel('From Joining Port')
        axs.set_ylabel('To Top 5 Leaving Ports')
        df8 = []
        df9 = pd.DataFrame()
        for b1, group in branch2:
      #print('Joining Port/Branch#1: ' + b1)
          f = group.b2.value_counts().nlargest(5).index
          group['b2_mask'] = group.b2.isin(f.tolist())
          group = group[group['b2_mask'] == True]
          group = group.drop_duplicates(subset=['b2'])
          group['path'] = group.index
          del group['b2_mask']
          del group['b1_mask']
          del group['path']
          x1= group['b1']
          y1= group['b2']
          plt.show()
          x1= group['b1']
          y1= group['b2']
      #axs.plot(x1, y1, marker='>', color='r', linestyle='none')
      
      #last part of assignment
          plt.scatter(group['b1'], group['b2'])

          df8.append(group)

          df10 = pd.concat(df8)
    
          df8 = df10
          df8 = df8.reset_index()
    
          df7['name'] = dfr.b2.unique()
          df7['id'] = df7.index
          name_id = pd.Series(df7.id.values,index=df7.name).to_dict()
          name_id['aberystwyth'] = 999
          df8['b1'] = df8['b1'].map(name_id)
          df8['b2'] = df8['b2'].map(name_id)
        
          df8['root'] = df8['root'].map(name_id)
          df8 = df8.dropna()
          x1= df8[['root','b1']]
          y1= df8['b2']
        
          x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.05)
      #^^ applying 'random_state='10'' will ensure the same results are processed each time
          clf = LinearRegression()
          clf.fit(x_train, y_train)
          results = clf.predict(x_test)
          results = results.astype(int)

          results = pd.Series(results)
          score1 = clf.score(x_train, y_train)
          nv_map = {v: k for k, v in name_id.items()}
          results = pd.Series(results).map(nv_map)
          b1_res = results
    #col two predict
          x1= df8[['root','b2']]
          y1= df8['b1']
          x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.05,random_state=10)
          clf = LinearRegression()
        
          clf.fit(x_train, y_train)
          results = clf.predict(x_test)
          score2 = clf.score(x_train, y_train)
          results = results.astype(int)
          results = pd.Series(results)
          results = pd.Series(results).map(nv_map)
          b2_res = results
    
    
          print('Estimated route of {}'.format(shipname))

          print('Root: ' + 'Aberystwyth')
          print('|_ Branch #1: ' + b2_res.values) # values were produced wrong way round
          print('  |__Branch #2: ' + b1_res.values)
          print('')
          print('Branch #1 Accuracy: ' + (str(score1)))
          print('Branch #2 Accuracy: ' + (str(score2)))
    
          print('To return to the ship menu, type "menu"')
          return_m = (input(': '))
          if return_m == 'menu':
              ship_menu()
          else:
              ship_menu()
    
    elif usr_in in ['exit', 'EXIT','Exit','e', 'E']:
        print('Program closing down...')
         #waits three seconds before closing the program down
        time.sleep(1)
    elif usr_in in ['menu', 'MENU','Menu','m', 'M']:
        command()
    else:
       print(f'{tcolours.HINT}remember you can use the first letter{tcolours.DEF}')
       ship_menu()
    


  
command()
