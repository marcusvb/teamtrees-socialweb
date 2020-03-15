import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dfd = pd.read_csv('data/donation_data/UpdatedDonationsperHour.csv')
dft = pd.read_csv('data/twitter_data/UpdatedTweetsperHour.csv')


offset = 300
l = len(dfd)


#TWEET SHIFT
tcorrelations = []
##############################################################################
#Shifted Correlations (Shifting Tweets)
for h in range(offset*2):    
    tcorrelations.append(np.corrcoef(dfd['av_rate'][offset:l-offset],\
                                    dft['count'][((offset*2)-h):(l-h)])[0][1])
    
    
#Tweet Shift PLOT
plt.figure(figsize=(9,6))
plt.xlabel('Tweet Shift (hours)',fontsize=20)
plt.ylabel('Correlation Coefficient',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.plot(range(-300,300),tcorrelations)
plt.grid()
##############################################################################



#DONATION SHIFT
dcorrelations = []
##############################################################################
#Shifted Correlations (Shifting Donations)
for h in range(offset*2):    
    dcorrelations.append(np.corrcoef(dft['count'][offset:l-offset],\
                                    dfd['av_rate'][((offset*2)-h):(l-h)])[0][1])
    
    
#Donation Shift PLOT
plt.figure(figsize=(9,6))
plt.xlabel('Donation Shift (hours)',fontsize=20)
plt.ylabel('Correlation Coefficient',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.plot(range(-300,300),dcorrelations)
plt.grid()
##############################################################################