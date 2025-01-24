"""
Created on Nov  5 10:24:06 2024

@author: shivang.sharma
"""
import pandas as pd


#File names
wthr='weather_data'
sls='prophet_output'
prm='promo_data'

#data prep

weather = pd.read_csv(path+'/'+wthr+'.csv')
sales = pd.read_csv(path+'/'+sls+'.csv')
promo = pd.read_csv(path+'/'+prm+'.csv')

m1 = sales.merge(weather, on = 'Date', how = 'left')

sales['Date'] = sales['Date'].astype(int)

m1['ID']= m1['Date'].astype(str)+'_'+ m1['Product_Key'].astype(str)

promo['ID']= promo['Date'].astype(str)+'_'+ promo['Product_Key'].astype(str)

m2 = m1.merge(promo, on = 'ID', how = 'left')

m2['Date'] = m2['Date_x'].astype(str)
m2['Year'] = m2['Date'].str[:4]
m2['Week'] = m2['Date'].str[-2:]
m2['Month'] = pd.DatetimeIndex(m2['ds']).month

del m2['Date_y']
del m2['ID']
del m2['Date_x']
del m2['Product_Key_y']

m2=m2.rename(columns={'Product_Key_x':'Product_Key'})

m3 = m2.copy()

m3.head()
m3.columns

m3 = m3.sort_values(['Product_Key','Date'])

m3['Sales_shift_4']=m3.groupby('Product_Key')['Sales'].shift(4)
m3['Sales_shift_4']=m3['Sales_shift_4'].fillna(method='bfill')

m3['Sales_shift_5']=m3.groupby('Product_Key')['Sales'].shift(5)
m3['Sales_shift_5']=m3['Sales_shift_5'].fillna(method='bfill')

m3['Sales_shift_6']=m3.groupby('Product_Key')['Sales'].shift(6)
m3['Sales_shift_6']=m3['Sales_shift_6'].fillna(method='bfill')

m3['Sales_shift_7']=m3.groupby('Product_Key')['Sales'].shift(7)
m3['Sales_shift_7']=m3['Sales_shift_7'].fillna(method='bfill')

m4 = m3.copy()

missing_df = m4.isnull().sum(axis=0).reset_index()

test1 = m4[['Week', 'W1']]
test1['W1'] = test1.groupby(['Week'])['W1'].ffill()

m4['W1'] = m4.groupby(['Week'])['W1'].ffill()
m4['W2'] = m4.groupby(['Week'])['W2'].ffill()
m4['W3'] = m4.groupby(['Week'])['W3'].ffill()
m4['W4'] = m4.groupby(['Week'])['W4'].ffill()
m4['W5'] = m4.groupby(['Week'])['W5'].ffill()
m4['W6'] = m4.groupby(['Week'])['W6'].ffill()
m4['W7'] = m4.groupby(['Week'])['W7'].ffill()
#
m4['Promo_shift_1']=m4.groupby('Product_Key')['Promo_Count'].shift(1)
m4['Promo_shift_1']=m4['Promo_shift_1'].fillna(method='bfill')

m4['Promo_shift_2']=m4.groupby('Product_Key')['Promo_Count'].shift(2)
m4['Promo_shift_2']=m4['Promo_shift_2'].fillna(method='bfill')

list(m4.columns)

m5=m4.drop(['trend_lower','trend_upper','seasonal_lower','seasonal_upper','seasonalities',
            'seasonalities_lower',
            'seasonalities_upper',
            'weekly_lower',
            'weekly_upper',
            'yearly_lower',
            'yearly_upper',
            'ds'],axis=1)
  
m5.to_csv(Dp_Op_path+'/'+'merged'+'.csv',index=False)
m5.to_csv(Dp_Op_path+'/'+'merged'+'.csv',index=False)
m5.isnull().sum()

m5.head()

