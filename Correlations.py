import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# read in the data
dfd = pd.read_csv('data/donation_data/UpdatedDonationsperHour.csv')
dft = pd.read_csv('data/twitter_data/count_sentiment_per_day_tweets.csv')
dfd_merged = pd.read_csv('data/donation_data/av_rate_per_day_donations_merged.csv')

# set periods of data you want
start_date = '2019-10-26'
end_date = '2020-02-26'
period_donations = (dfd_merged['date'] >= start_date) & (dfd_merged['date'] <= end_date)
period_social = (dft['date'] >= start_date) & (dft['date'] <= end_date)

# select data
dfd_merged = dfd_merged.loc[period_donations]
dft = dft.loc[period_social]

# make dates a date time format
dfd['date'] = pd.to_datetime(dfd['date'], infer_datetime_format=True)
dft['date'] = pd.to_datetime(dft['date'], infer_datetime_format=True)
dfd_merged['date'] = pd.to_datetime(dfd_merged['date'], infer_datetime_format=True)

# print("This is donation data: \n", dfd_merged)

offset = 8
l = len(dfd_merged)


#TWEET SHIFT
tcorrelations = []
tcorrelations_positive = []
tcorrelations_negative = []
tcorrelations_neutral = []
##############################################################################
#Shifted Correlations (Shifting Tweets)
for h in range(offset*2):    
    tcorrelations.append(np.corrcoef(dfd_merged['av_rate'][offset:l-offset],\
                                    dft['count'][((offset*2)-h):(l-h)])[0][1])
for h in range(offset*2):
    tcorrelations_positive.append(np.corrcoef(dfd_merged['av_rate'][offset:l-offset],\
                                    dft['n_positive'][((offset*2)-h):(l-h)])[0][1])
for h in range(offset*2):
    tcorrelations_negative.append(np.corrcoef(dfd_merged['av_rate'][offset:l-offset],\
                                    dft['n_negative'][((offset*2)-h):(l-h)])[0][1])
for h in range(offset*2):
    tcorrelations_neutral.append(np.corrcoef(dfd_merged['av_rate'][offset:l-offset],\
                                    dft['neutral'][((offset*2)-h):(l-h)])[0][1])
    
    
#Tweet Shift PLOT
plt.figure(figsize=(9,6))
plt.xlabel('Tweet Shift (day)',fontsize=20)
plt.ylabel('Correlation Coefficient',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.plot(-1* np.arange(-offset,offset),tcorrelations, label='total', linewidth=0.7)
plt.plot(-1* np.arange(-offset,offset),tcorrelations_positive, 'g-', label='positive', linewidth=0.7)
plt.plot(-1* np.arange(-offset,offset),tcorrelations_negative, 'r-', label='negative', linewidth=0.7)
plt.plot(-1* np.arange(-offset,offset),tcorrelations_neutral, 'y-', label='neutral', linewidth=0.7)
plt.legend()

##############################################################################



#DONATION SHIFT
# dcorrelations = []
##############################################################################
#Shifted Correlations (Shifting Donations)
# for h in range(offset*2):
#     dcorrelations.append(np.corrcoef(dft['count'][offset:l-offset],\
#                                     dfd_merged['av_rate'][((offset*2)-h):(l-h)])[0][1])
    
    
# #Donation Shift PLOT
# plt.figure(figsize=(9,6))
# plt.xlabel('Donation Shift (hours)',fontsize=20)
# plt.ylabel('Correlation Coefficient',fontsize=20)
# plt.tick_params(axis='both', which='major', labelsize=15)
# plt.plot(range(-300,300),dcorrelations)
# plt.grid()
##############################################################################
plt.show()