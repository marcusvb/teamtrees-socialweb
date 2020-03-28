import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

register_matplotlib_converters()

sns.set()

plt.style.use('ggplot')
sns.set_palette("hls")
plt.rcParams.update({'font.size': 16})

# # pd.set_option('display.max_colwidth', -1)
# # pd.set_option('display.max_rows', 500)
# # pd.set_option('display.max_columns', 500)
# # pd.set_option('display.width', 150)

def get_tweet_data():
    df = pd.read_csv("data/twitter_data/tweets.csv", delimiter=";", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    return df

def get_donation_data():
    df = pd.read_csv('data/donation_data/parsed-team-trees-10second.csv', header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    return df

#CLUSTERING CODE

def get_tweet_data(url):
    df = pd.read_csv(url, delimiter=";", header=0)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    return df



# Return dataframe that has tweets based on groups instead of on clusters
# Groupnames should contain list of stringnames of groups
# Cpg should contain lists with all the clusternumbers per group in groupnames
def grouping(Data, Groupnames, Clusters_per_group):
    tweets_per_group = []

    for i, clustgroup in enumerate(Clusters_per_group):

        for c in clustgroup:

            tweets = list(Data.loc[Data['cluster'] == c]['tweetlist'])
            for t in tweets:

                words = t.lower().strip().split(' ')  #id is the last word in tweet
                groupi = Groupnames[i]

                #some groups that were identified manually

                if 'nikon' in t or 'Nikon' in t or 'Canon PowerShot' in t:
                    groupi = 'Advertisements'

                elif 'elonmusk' in t or 'BillGates' in t or 'JeffBezos' in t:
                    groupi = 'Celebrity related'


                tweets_per_group.append([groupi, t, words[-1]])

    tweets_per_group = pd.DataFrame(tweets_per_group)
    return tweets_per_group

# Divide set of tweets in half because it is not managable to perform k-means on a set of 100000+ tweets

df = pd.read_csv('clusterfile')
df.columns = ['cluster', 'tweetlist']

df1 = df[60000 :]
df2 = df[: 60000]

# Code for primal investigation of clusters


# plt.figure()
# sns.distplot(df1['cluster'], kde=False)
# sns.distplot(df2['cluster'], kde=False)
# plt.show()


# for i in range(21):
#     print('Cluster number: {}, frequency: '.format(i), sum((df1['cluster']==i)))
#
# for i in range(21):
#     print(sum((df2['cluster']==i)))
#     # print('Cluster number: {}, frequency: '.format(i), sum((df2['cluster']==i)))


# tlists = []
#
# for i in range(21):
#     tweets = list(df2.loc[df2['cluster'] == i]['tweetlist'])
#     tlist = []
#     for tweet in tweets:
#         words = tweet.lower().strip().split(' ')
#         tlist.extend(words[:-1])
#     tlists.append(tlist)
#
# for t in tlists:
#     C = Counter(t)
#     # print(C.most_common(20))



#Code for grouping after primal investigation is done

Groups = ['General hype', 'Stimulating others', 'MrBeast + Youtube', 'People donating themselves', 'Celebrity related', 'Advertisements']

#these lists were created by manual inspection of the tweets in each cluster, Each list of clusters comes across with the names in Groups
clusters_per_group =  [[6,8,17], [1,2,3,12,19,20],[0, 9, 14],[13], [3,4], [5]]

GroupData = grouping(df1, Groups, clusters_per_group)


GroupData.columns = ['Group', 'Tweet', 'id']

clusters_per_group2 =  [[1,2,7,10,15,17,19],[0,3,4,5,6,11,14,18,20],[8,9, 12,13],[16],[16],[16]]


GroupData2 = grouping(df2, Groups, clusters_per_group2)
GroupData2.columns = ['Group', 'Tweet', 'id']





#Code for adding group to files in original dataframe
data_url = "data/twitter_data/tweets.csv"
OriginalData = get_tweet_data(data_url)

# OriginalData['Groups'] = 0
OriginalData['id'] = OriginalData['id'].astype(str)

#merging the groups with the original dataframe
Mergedf1 = OriginalData.merge(GroupData, how = 'inner', on = 'id', sort = True)


Mergedf2 = OriginalData.merge(GroupData2, how = 'inner', on = 'id', sort = True)


Mergedf = pd.concat([Mergedf1, Mergedf2])
Mergedf.to_csv('Groupdataframe')

# Mergedf = pd.read_csv('Groupdataframe', delimiter=",", header=0)
# Mergedf.columns = ['none','username','date','retweets','favorites','text','geo','mentions','hashtags','Groups','id','Group','permalink']

def plot_cluster_data(tweet_dfs=Mergedf, groupnames = Groups):

    # print(Mergedf['Group'])

    # read in data per time unit

    # plots
    # plt.bar(donation_df['date'], donation_df['raised_capital'])
    # plt.xlabel("Date")
    # plt.ylabel("Raised capital ($)")

    fig, ax1 = plt.subplots(figsize = (10,10))
    ax1.set_xlabel('Date')

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    datalist = []

    for i, group in enumerate(groupnames):
        print(group)
        color = colors[i]

        tweet_df = tweet_dfs[tweet_dfs['Group'] == group]

        ax1.set_ylabel('Frequency of Tweets')
        ax1.tick_params(axis='y')

        datalist.append(tweet_df['date'])
        plt.hist(tweet_df['date'], alpha=0.5, density = True, bins = 20, color=color, label = group, histtype='step', linewidth=1.5)




    plt.legend()

    xticklist = list(tweet_df['date'])
    start = xticklist[0]
    end = xticklist[-1]
    ticks = [start, start+(end-start)/3, start+(end-start)*2/3, end]
    ax1.set_xticks(ticks)

    # Zoomin on two peaks
    axins = zoomed_inset_axes(ax1, 2.25, loc='right')  # zoom-factor: 5, location: upper-left

    for i, dat in enumerate(datalist):
        plt.hist(dat, alpha=0.5, density = True, bins = 20, color=colors[i], histtype='step', linewidth=1.5)
    axins.set_xlim([start + (end - start) * 2.5/ 10, start + (end - start) / 2])  # apply the x-limits
    axins.set_ylim([0, 0.02])  # apply the y-limits
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    mark_inset(ax1, axins, loc1=4, loc2=2, fc="none", ec="1.5")

    fig.tight_layout()
    plt.savefig('Clusteringplot_zoomin')
    plt.show()


plot_cluster_data()

