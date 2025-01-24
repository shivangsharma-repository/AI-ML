#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 25 21:14:53 2024

@author: shivang sharma
"""

# -*- coding: utf-8 -*-
"""
Created on Nov 25 21:14:53 2024

@author: shivang.sharma
"""

#Packages

import pandas as pd
from fbprophet import Prophet

#data prep for prophet

#Name

sls='sales_data_v2'
mapping='Date_mapping'

#Read file
d1 = pd.read_excel(Ip_path+'/'+sls+'.xlsx')

date_map=pd.read_excel(Ip_path+'/'+mapping+'.xlsx')

d1[d1['Date']==201553]

d1['Date'].unique()

date_map['ds']=pd.to_datetime(date_map['ds'])

d1=d1.merge(date_map,on='Date',how='left')

d1['ID']=d1['ds'].astype(str) + "_" + d1['Product_Key'].astype(str)

d1 = d1.sort_values(['Product_Key', 'ds']).reset_index(drop=True)
#
d2=d1.loc[d1['Date']<201648,:].reset_index(drop=True)
d2.shape
d2.tail()

#Model1
js=[]
for i in d2['Product_Key'].unique():
    d11 = d2[['Product_Key', 'ds', 'Sales']]
    d11.columns=['PK', 'ds', 'y']
    d4 = d11[d11['PK']== i]
    d5 = d4[['ds','y']]
    m = Prophet(weekly_seasonality = True,mcmc_samples=True)
    m.fit(d5)
    future = m.make_future_dataframe(periods = 5, freq = 'W')
    forecast = m.predict(future)
    forecast["PK"] = i
    js.append(forecast)    
    
res = pd.concat(js)
res.shape

#Model 2
js1=[]
for i in d2['Product_Key'].unique():
    d11 = d2[['Product_Key', 'ds', 'Sales']]
    d11.columns=['PK', 'ds', 'y']
    d4 = d11[d11['PK']== i]
    d5 = d4[['ds','y']]
    m = Prophet(weekly_seasonality = True, growth='linear')
    m.fit(d5)
    future = m.make_future_dataframe(periods = 5, freq = 'W')
    forecast = m.predict(future)
    forecast["PK"] = i
    js1.append(forecast)
    
res1 = pd.concat(js1)

res['ID'] = res['ds'].astype(str) + "_" + res['PK'].astype(str)
res1['ID'] = res1['ds'].astype(str) + "_" + res1['PK'].astype(str)

res=res[['ID','yhat']]

a = res.merge(res1, on = 'ID', how = 'left')

a['yhat'] = 0.9*a['yhat_x'] + 0.1*a['yhat_y']

a['yhat_final']=0.78*a['yhat']
a.loc[a['yhat_final']<0,['yhat_lower','yhat_final']]=0

a['ds']=pd.to_datetime(a['ds'])

a=a.merge(date_map,on='ds',how='left')

a=a.merge(d1,on='ID',how='left')

a=a.loc[a['Date_y']!=201648,:]

a=a.drop(['Date_y','ds_y','ID','yhat','yhat_x','yhat_y','Product_Key'],axis=1)

a=a.rename(columns={'Date_x':'Date','PK':'Product_Key','ds_x':'ds'})

a.to_csv(Op_path+'/'+'prophet_output'+'.csv',index=False)
a.to_csv(Dp_Ip_path+'/'+'prophet_output'+'.csv',index=False)
