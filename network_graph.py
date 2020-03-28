# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 16:25:18 2020
"""

# Imports.
import tweepy 
import csv
import pandas as pd
import re
import time
import numpy as np

# Twitter API credentials.
consumer_key = 'test'
consumer_secret = 'test'
access_key = 'test'
access_secret = 'test'

# Check if authorized already.
try:
    auth
except NameError:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
      
    
def get_screennames():
    """
    Extracts screennames from tweet link.
    """
    
    # Opening and reading the file, initializing array to store data.
    df = pd.read_csv("data/twitter_data/tweets.csv", delimiter=";", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    screennames = np.array([['test1', 1222413921874849792, 74, 338, 169, 420, 1996], 
                             ['test2', 1222413921874849792, 74, 338, 169, 420, 1996]])
    
    # Iterate through file.
    for link, date in zip(df['permalink'], df['date']):
        start = time.time()
        
        if not isinstance(link, str) :
            pass
        else:
            
            # Get the screen name.
            name = re.search('https://twitter.com/(.*)/status/', str(link)).group(1)
            
            # Check if not extacted already from an earlier tweet.
            if not name in screennames[:,0]:
                
                # Skip users that deleted their account or set to private.
                try:
                    # Get user instance from twitter API
                    user = api.get_user(name)
                    new_el = [name, user.id, 
                               user.followers_count, user.friends_count, 
                               user.statuses_count, user.created_at, date]
                    
                    screennames = np.append(screennames, [new_el], axis=0)
            
                    # Write to csv.
                    with open('data/twitter_data/tweeter_info.csv', 'a') as f:
                        writer = csv.writer(f, delimiter=';')
                        writer.writerow(new_el) 
                    
                    # Staying within Twitter's rate limit.
                    while time.time() - start < 1:
                        time.sleep(0.00001)
                        
                except:
                    print("user not found")
    
    return screennames



def create_edge_file(datafrm):
    """
    Writes an adjacency file.
    """
    
    # Create list of all the id's in the original file.
    start = time.time()
    list_of_uids = [int(i) for i in datafrm['id']]
    
    # Iterate through the original file
    for index, row in datafrm.iterrows():
        
        # Keep track of process.
        print("Start: ", index, row['id'], time.time() - start)
            
        # Skip users with over 50,000 followers.
        if row['followers'] > 50000:
            print("Skipped user: ", row['screen_name'])
            
        # Skip users with 0 followers.
        elif row['followers'] > 0: 
            
            # Skip users that deleted their account or set to private.
            try:
                
                out = get_followers(row['screen_name'], row['id'])
                out = [i for i in out if i in list_of_uids]
    
            	# Write the csv.
                with open('data/twitter_data/corrected_edge_list.csv', 'a') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow(out)
        
            except:
                print("Error on user: ", row['screen_name'])



def get_followers(scrn_nm, uid):
    """
    Creates a list containing shortened user_id's of all followers 
    for the given screen_name. 
    """
    
    # Initialize list to fill with followers
    followers = [uid]
       
    # Iterate through the pages, one page per minute to stay within Twitter's rate limit
    for page in tweepy.Cursor(api.followers_ids, screen_name=scrn_nm).pages(): 
        start = time.time()
        
        for i in page:
            followers.append(int(str(i)[:12]))
            
        while time.time() - start < 60:
            time.sleep(0.00001)
        
    return followers