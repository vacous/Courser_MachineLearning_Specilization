# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 08:06:36 2016

@author: Administrator
"""

import pandas as pd
import datetime 
import pandas.io.data as web 
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np

"""
Example 01 
"""
# stock price plot 
#style.use('ggplot')
#start = datetime.datetime(2010,1,1)
#end = datetime.datetime(2015,1,1)
#df = web.DataReader("XOM", "yahoo", start, end)
#print(df.head())
#df['Adj Close'].plot()
#plt.show()

"""
Example 02 
"""
# data frame = dictionary 
web_stats = {'day': [1,2,3,4,5,6],'visitors':[43,44,45,46,47,48],'Bounce_rate': [1,2,3,4,5,6]}
df = pd.DataFrame(web_stats)
#print(df)
#print(df.head())
#print(df.tail())
##print(df.tail(2))
#print(df.set_index('day')) # return a new dataframe 
#df = df.set_index('day')
## or
#df.set_index('day', inplace = True)           
#print(df.head())

## reference specific column 
print(df['visitors'])
print(df.visitors) # same 
print(df[['visitors','Bounce_rate']]) # print two columns 

test = df[['visitors','Bounce_rate']]
test = np.array(test) # convert to np.array 
print(test) 
print(pd.DataFrame(test)) # convert back 

