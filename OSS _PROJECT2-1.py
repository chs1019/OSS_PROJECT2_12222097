#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import csv
data=pd.read_csv('./2019_kbo_for_kaggle_v2.csv')
classic_data=data[(data['year'] >= 2015) & (data['year'] <= 2018)]
top_10_players={}
for year in range(2015, 2019):
    top_10_players[year]={
        'H':classic_data.nlargest(10,'H'),  
        'avg':classic_data.nlargest(10,'avg'),
        'HR':classic_data.nlargest(10,'HR'),
        'OBP':classic_data.nlargest(10,'OBP')
    }

for year in top_10_players.items(): 
        print(f'{year}')
print('done')

saber_data=data[(data['year']==2018)]
war_player={}
pos=['포수','1루수','2루수','3루수','유격수','좌익수','중견수','우익수'] 
for pos in pos:
    war_player[pos]=saber_data[saber_data['cp']==pos].nlargest(1,'war')
for pos in war_player.items():
    print(f'{pos}')
print('done')

columns=['R','H','HR','RBI','SB','war','avg','OBP','SLG']
cor=data[columns].corrwith(data['salary'].abs())
cor_top=cor.idxmax()
print(f'가장 상관관계가 높은것:{cor_top}')
print('done')


# In[ ]:





# In[ ]:




