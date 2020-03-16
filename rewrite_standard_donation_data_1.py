import pandas as pd

dft = pd.read_csv('data/donation_data/team-tree-donation-data.csv', header=0, delimiter="\t")
print(dft)

new_dft = pd.DataFrame(columns=['date', 'amount', 'raised_capital', 'rate_of_funding'])

new_dft['date'] = dft['date']
new_dft['raised_capital'] = dft['raised_capital']
# new_dft['rate_of_funding'] = dft['rate_of_funding']

new_dft['rate_of_funding'] = dft['rate_of_funding'].apply(lambda x: float(x.strip('/min')))

new_dft.to_csv('data/donation_data/normal_data_rewritten_as_parsed_10_sec_data', header=True)
