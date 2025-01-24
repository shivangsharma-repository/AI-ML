#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Dec 5 17:50:58 2024

@author: shivang.sharma
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Dec 5 17:50:58 2024

@author: shivang.sharma
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV

#Path


#Name

ip_file='merged'
keys_zero='Keys_zero_sales_v2'

#Input file

df=pd.read_csv(path+'/'+ip_file+'.csv')

#Based on last 2 months of data list of keys which have zero sales from 201640-201647

df_key_zero_sales=pd.read_excel(path+'/'+keys_zero+'.xlsx')

keys_zero=list(df_key_zero_sales['Fus'])

#df2=df.drop('yhat_final',axis=1)

#Train test split

df_train=df.loc[df['Date']<201649,:].reset_index(drop=True)
df_test=df.loc[df['Date']>=201649,:].reset_index(drop=True)

#Divide into X and Y

X_train=df_train.loc[:,df_train.columns!='Sales']
y_train=np.array(df_train.loc[:,'Sales'].values)

X_test=df_test.loc[:,df_test.columns!='Sales']
y_test=np.array(df_test.loc[:,'Sales'].values)

#List of FU's

prod_key=['FGB0748','FGB0737','FGB6596','FGB0727','FGB6299','FGB0726','FGB0735','FGB0723','FGB6543','FGB0754']

#Parametrs dict

###Parameters
##
FU_params={"FGB0748":{"objective":"reg:linear","eta":0.75,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB0751":{"objective":"reg:linear","eta":0.11,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB0737":{"objective":"reg:linear","eta":0.736,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB6596":{"objective":"reg:linear","eta":0.007,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB0727":{"objective":"reg:linear","eta":0.01,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB6299":{"objective":"reg:linear","eta":0.003,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB6542":{"objective":"reg:linear","eta":0.008,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB0726":{"objective":"reg:linear","eta":0.003,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB0723":{"objective":"reg:linear","eta":0.2,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB6543":{"objective":"reg:linear","eta":0.4,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB0754":{"objective":"reg:linear","eta":0.05,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
    "FGB0735":{"objective":"reg:linear","eta":0.01,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1}

      }
##

FU_nr={"FGB0748":20,
       "FGB0751":13,
       "FGB0737":15,
       "FGB6596":2800,
       "FGB0727":800,
       "FGB6299":2500,
       "FGB6542":500,
       "FGB0726":1500,
       "FGB0723":35,
       "FGB6543":40,
       "FGB0754":300,
       "FGB0735":1000
      }

##Other set of parameters tried

#Set1

#FU_params={"FGB0748":{"objective":"reg:linear","eta":0.055,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0751":{"objective":"reg:linear","eta":0.0035,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0737":{"objective":"reg:linear","eta":0.01,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB6596":{"objective":"reg:linear","eta":0.007,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0727":{"objective":"reg:linear","eta":0.01,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB6299":{"objective":"reg:linear","eta":0.003,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB6542":{"objective":"reg:linear","eta":0.008,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0726":{"objective":"reg:linear","eta":0.003,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0723":{"objective":"reg:linear","eta":0.2,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB6543":{"objective":"reg:linear","eta":0.4,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0754":{"objective":"reg:linear","eta":0.05,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0735":{"objective":"reg:linear","eta":0.01,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1}
#
#      }
##
#FU_nr={"FGB0748":150,
#       "FGB0751":460,
#       "FGB0737":1200,
#       "FGB6596":2800,
#       "FGB0727":800,
#       "FGB6299":2500,
#       "FGB6542":500,
#       "FGB0726":1500,
#       "FGB0723":35,
#       "FGB6543":40,
#       "FGB0754":300,
#       "FGB0735":1000
#      }

#Set2


#FU_params={"FGB0748":{"objective":"reg:linear","eta":0.7,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0751":{"objective":"reg:linear","eta":0.15,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0737":{"objective":"reg:linear","eta":0.7,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB6596":{"objective":"reg:linear","eta":0.5,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0727":{"objective":"reg:linear","eta":0.35,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB6299":{"objective":"reg:linear","eta":0.1,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB6542":{"objective":"reg:linear","eta":0.7,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0726":{"objective":"reg:linear","eta":0.15,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0723":{"objective":"reg:linear","eta":0.8,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB6543":{"objective":"reg:linear","eta":0.4,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0754":{"objective":"reg:linear","eta":0.05,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1},
#    "FGB0735":{"objective":"reg:linear","eta":0.01,"min_child_weight":1,"subsample":0.7,"colsample_bytree":0.6,"silent":1,"max_depth":6,"seed":1}
#
#      }
##
#FU_nr={"FGB0748":25,
#       "FGB0751":15,
#       "FGB0737":35,
#       "FGB6596":15,
#       "FGB0727":15,
#       "FGB6299":300,
#       "FGB6542":15,
#       "FGB0726":80,
#       "FGB0723":15,
#       "FGB6543":40,
#       "FGB0754":300,
#       "FGB0735":1000
#      }




##XGBOOST model
#
##
def runXGB(train_X, train_y,params,nr):
    xgtrain = xgb.DMatrix(train_X, label=train_y)
#    xgtest = xgb.DMatrix(test_X)
    model = xgb.train(params, xgtrain,nr)
    return model
####
df_test_2=df_test.copy()
##
##Final Predictions 
#                         
for prod in prod_key:
    df_train_1=df_train.loc[df_train['Product_Key']==prod,:].reset_index(drop=True)
    X_train=df_train_1.loc[:,df_train_1.columns!='Sales']
    y_train=np.array(df_train_1.loc[:,'Sales'].values)
    X_train_1=X_train.drop(['Product_Key','Date','yhat_final'],axis=1)    
    model = runXGB(X_train_1,y_train,FU_params[prod],FU_nr[prod])
    df_test_1=df_test.loc[df_test['Product_Key']==prod,:].reset_index(drop=True)
    X_test=df_test_1.loc[:,df_test_1.columns!='Sales']
    X_test_1=X_test.drop(['Product_Key','Date','yhat_final'],axis=1)    
    y_pred = model.predict(xgb.DMatrix(X_test_1))
    df_test_2.loc[df_test_2['Product_Key']==prod,'XGB_Pred']=y_pred
            

df_test_2['XGB_Pred']=df_test_2['XGB_Pred'].fillna(df_test_2['yhat_final'])
#
##
#
df_test_2.loc[df_test_2['Sales']==0,'XGB_Pred']=0
#
df_test_2.loc[df_test_2['Product_Key'].isin(keys_zero),'XGB_Pred']=0
#
accuracy=1-(sum(abs(df_test_2['XGB_Pred']-df_test_2['Sales']))/sum(df_test_2['Sales']))
##
print(accuracy)

##
df_test_2.to_excel(Op_Path+'/'+'Final_Output'+'.xlsx',index=False)

xgb.plot_importance(model)


##For tuning
##
#def runXGB(train_X, train_y,l,n):
#    params={}
#    params["objective"]="reg:linear"
#    params["eta"]=l
#    params["min_child_weight"]=1
#    params["subsample"]=0.7
#    params["colsample_bytree"]=0.6
#    params["silent"]=1
#    params["max_depth"]=6
#    params["seed"]=1
#    plst=list(params.items())
#    xgtrain = xgb.DMatrix(train_X, label=train_y)
##    xgtest = xgb.DMatrix(test_X)
#    model = xgb.train(plst,xgtrain,n)
#    
#    return model
#
#
#lr=[0.001,0.003,0.005,0.008,0.01,0.02,0.03,0.04,0.05,0.055,0.06,0.07,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7]
#nr=[10,15,20,25,30,35,40,45,50,60,100,200,250,300,350,400,450,500,600,700,800,900,1000,1500,2000,2500]
#
#lr=[0.45,0.5,0.6,0.7]
#nr=[5,10,15,20,25,30,35,40,45]


##md=[2,4,6,8,10]

#To tune the parameters

#error={}
#for i in lr:
#    for j in nr:
##        for k in md:
#        for prod in prod_key:
#            print('For Lr %f',i)
#            print('For nr %f',j)
##            print('For depth %d',k)
#            df_train_1=df_train.loc[df_train['Product_Key']==prod,:].reset_index(drop=True)
#            X_train=df_train_1.loc[:,df_train_1.columns!='Sales']
#            y_train=np.array(df_train_1.loc[:,'Sales'].values)
#            X_train_1=X_train.drop(['Product_Key','Date','yhat_final'],axis=1)    
#            model = runXGB(X_train_1,y_train,i,j)
#            df_test_1=df_test.loc[df_test['Product_Key']==prod,:].reset_index(drop=True)
#            X_test=df_test_1.loc[:,df_test_1.columns!='Sales']
#            X_test_1=X_test.drop(['Product_Key','Date','yhat_final'],axis=1)    
#            y_pred = model.predict(xgb.DMatrix(X_test_1))
#            
#            y_test_prod=np.array(df_test_2.loc[df_test_2['Product_Key']==prod,'Sales'])
#            
#            error[(i,j)]=sum(abs(y_test_prod-y_pred))
#        #           
#         
##            accuracy[(i,j)]=1-(sum(error)/sum(y_test_prod))
#
##print(list(error))            
##
#minimum=min(error,key=error.get)            
#
#print(minimum,error[minimum])
#

#xgboost cross validation(Used sklearn wrapper of xgb for cross validation)

#def crossvalXGB(train_X, train_y):
#    params={}
##    params["objective"]="reg:linear"
#    params["learning_rate"]=[0.1,0.2]#[0.01,0.03,0.04,0.05,0.06,0.07,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
##    params["min_child_weight"]=range(1,10)
##    params["subsample"]=[0.5,0.6,0.7,0.8]
##    params["colsample_bytree"]=0.6  #[0.4,0.5,0.6,0.7]
##    params["silent"]=1
##    params["max_depth"]= #range(3,10)
##    params["seed"]=1
##    params['n_jobs']=4             
##    params['n_estimators']=[10,15,20,25,30,35,40,45,50,60,100,200,250,300,350,400,450,500]
#    model=XGBRegressor(colsample_bytree=0.6,seed=1,n_estimators=2)
#    gsearch=GridSearchCV(estimator=model,param_grid=params,scoring='neg_mean_absolute_error',cv=2,n_jobs=-1,iid=False)
#    gsearch.fit(train_X,train_y)  
#          
#    return gsearch.best_params_, gsearch.best_score_    
##
#
#for prod in prod_key:
#    df_train_1=df_train.loc[df_train['Product_Key']==prod,:].reset_index(drop=True)
#    X_train=df_train_1.loc[:,df_train_1.columns!='Sales']
#    y_train=np.array(df_train_1.loc[:,'Sales'].values)
#    X_train_1=X_train.drop(['Product_Key','Date','yhat_final'],axis=1)    
#    best_params,best_score = crossvalXGB(X_train_1,y_train)
#    print('Prod : ',prod)
#    print('Params : ',best_params)
#    print('Best Score : ',best_score )
#    
#    
