import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

dfd = pd.read_csv('data/donation_data/UpdatedDonationsperHour.csv')
dft = pd.read_csv('data/twitter_data/count_sentiment_per_hour_tweets.csv')
dfd_merged = pd.read_csv('data/donation_data/av_rate_per_hour_donations_merged.csv')



start_date = '2019-10-26 15:00'
end_date = '2020-02-26 14:00'
period_donations = (dfd_merged['date'] >= start_date) & (dfd_merged['date'] <= end_date)
period_social = (dft['date'] >= start_date) & (dft['date'] <= end_date)

dfd_merged = dfd_merged.loc[period_donations]
dft = dft.loc[period_social]

dfd['date'] = pd.to_datetime(dfd['date'], infer_datetime_format=True)
dft['date'] = pd.to_datetime(dft['date'], infer_datetime_format=True)
dfd_merged['date'] = pd.to_datetime(dfd_merged['date'], infer_datetime_format=True)

print("This is donation data: \n", dfd_merged)
print("This is social data: \n", dft)

# assert print("it stops here!")

offset = 300
l = len(dfd_merged)


#TWEET SHIFT
tcorrelations = []
##############################################################################
#Shifted Correlations (Shifting Tweets)
for h in range(offset*2):    
    tcorrelations.append(np.corrcoef(dfd_merged['av_rate'][offset:l-offset],\
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
                                    dfd_merged['av_rate'][((offset*2)-h):(l-h)])[0][1])
    
    
#Donation Shift PLOT
plt.figure(figsize=(9,6))
plt.xlabel('Donation Shift (hours)',fontsize=20)
plt.ylabel('Correlation Coefficient',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.plot(range(-300,300),dcorrelations)
plt.grid()
##############################################################################
plt.show()