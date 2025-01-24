import pandas as pd
import numpy as np
from sklearn.preprocessing import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import model_selection

def one_hot_mat_mul(data,cols):
    df = data[cols]
    df1 = pd.get_dummies(df)

    x = np.repeat(df1.values,np.shape(df1)[1],axis=1)
    y = np.tile(df1.values,np.shape(df1)[1])
    z = x*y
    colnames = [x+y for x, y in zip(np.repeat(df1.columns, np.shape(df1)[1]),          np.tile(df1.columns, np.shape(df1)[1]))]

    final_data = pd.DataFrame(z, columns=colnames)
    d2 = pd.concat([data,final_data],ignore_index=True,axis=1)
    return d2

def mul_interaction(data,cols):
    df = data[cols]
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].replace('Others',999)
    x = np.repeat(df.values,np.shape(df)[1],axis=1)
    y = np.tile(df.values,np.shape(df)[1])
    z = x*y
    colnames = [x+y for x, y in zip(np.repeat(df.columns, np.shape(df)[1]),     np.tile(df.columns, np.shape(df)[1]))]
    final_data = pd.DataFrame(z, columns=colnames)
    d2 = pd.concat([data,final_data],ignore_index=True,axis=1)
    return d2

def add_interaction(data,cols):
    df = data[cols]
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].replace('Others',999)
    x = np.repeat(df.values,np.shape(df)[1],axis=1)
    y = np.tile(df.values,np.shape(df)[1])
    z = x+y
    colnames = [x+y for x, y in zip(np.repeat(df.columns, np.shape(df)[1]),     np.tile(df.columns, np.shape(df)[1]))]
    final_data = pd.DataFrame(z, columns=colnames)
    d2 = pd.concat([data,final_data],ignore_index=True,axis=1)
    return d2

def div_interaction(data,cols):
    df = data[cols]
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].replace('Others',999)
    x = np.repeat(df.values,np.shape(df)[1],axis=1)
    y = np.tile(df.values,np.shape(df)[1])
    z = x/y
    colnames = [x+y for x, y in zip(np.repeat(df.columns, np.shape(df)[1]),     np.tile(df.columns, np.shape(df)[1]))]
    final_data = pd.DataFrame(z, columns=colnames)
    d2 = pd.concat([data,final_data],ignore_index=True,axis=1)
    return d2

def minus_interaction(data,cols):
    df = data[cols]
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].replace('Others',999)
    x = np.repeat(df.values,np.shape(df)[1],axis=1)
    y = np.tile(df.values,np.shape(df)[1])
    z = x-y
    colnames = [x+y for x, y in zip(np.repeat(df.columns, np.shape(df)[1]),     np.tile(df.columns, np.shape(df)[1]))]
    final_data = pd.DataFrame(z, columns=colnames)
    d2 = pd.concat([data,final_data],ignore_index=True,axis=1)
    return d2


def double_mean(train2, val2, col=['LOSS_POSTCODE', 'MNTH_OCC_DT'], target='CAT_FLAG'):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
    t1 = pd.DataFrame(columns=col)
    for i, (train_index, val_index) in enumerate(kf.split(train2)):
        x_tr, x_val = train2.iloc[train_index], train2.iloc[val_index]
        t = x_tr.groupby(col)[target].mean().reset_index()
        name = str(col[0]) + '_' + str(col[1]) + '_' + str(target) + '_mean'
        t = t.rename(columns={target: name})
        x_val = pd.merge(x_val, t, on=col, how='left')
        test_encoding = pd.merge(val2[col], t, on=col, how='left')
        t1['test_encoding' + '_' + str(i)] = test_encoding[name]
        # t1['test_encoding'+'_'+str(i)] = t['loss_month_mean']
        if i == 0:
            train_new = x_val
        else:
            train_new = pd.concat([train_new, x_val])

    val2[name] = t1.mean(axis=1)

    return train_new, val2


def mean_encoding_single(x_train, x_test, col='work_type', target='stroke'):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
    t1 = pd.DataFrame()
    for i, (train_index, val_index) in enumerate(kf.split(x_train)):
        x_tr, x_val = x_train.iloc[train_index], x_train.iloc[val_index]
        p_class_encoding = x_val[col].map(x_tr.groupby([col])[target].mean())
        x_val[col + '_mean_encoding_' + target] = p_class_encoding
        test_encoding = x_test[col].map(x_tr.groupby([col])[target].mean())
        t1['test_encoding' + '_' + str(i)] = test_encoding
        if i == 0:
            train_new = x_val
        else:
            train_new = pd.concat([train_new, x_val])
    x_test[col + '_mean_encoding_' + target] = t1.mean(axis=1)
    return train_new, x_test


def median_encoding_single(x_train, x_test, col='work_type', target='stroke'):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
    t1 = pd.DataFrame()
    for i, (train_index, val_index) in enumerate(kf.split(x_train)):
        x_tr, x_val = x_train.iloc[train_index], x_train.iloc[val_index]
        p_class_encoding = x_val[col].map(x_tr.groupby([col])[target].median())
        x_val[col + '_median_encoding_' + target] = p_class_encoding
        test_encoding = x_test[col].map(x_tr.groupby([col])[target].median())
        t1['test_encoding' + '_' + str(i)] = test_encoding
        if i == 0:
            train_new = x_val
        else:
            train_new = pd.concat([train_new, x_val])
    x_test[col + '_median_encoding_' + target] = t1.mean(axis=1)

    return train_new, x_test


def first_quantile_encoding_single(x_train, x_test, col='work_type', target='stroke'):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
    t1 = pd.DataFrame()
    for i, (train_index, val_index) in enumerate(kf.split(x_train)):
        x_tr, x_val = x_train.iloc[train_index], x_train.iloc[val_index]
        p_class_encoding = x_val[col].map(x_tr.groupby([col])[target].quantile(.25))
        x_val[col + '_1quantile_encoding_' + target] = x_val[target] / p_class_encoding
        test_encoding = x_test[col].map(x_tr.groupby([col])[target].quantile(.25))
        t1['test_encoding' + '_' + str(i)] = test_encoding
        if i == 0:
            train_new = x_val
        else:
            train_new = pd.concat([train_new, x_val])
    x_test[col + '_1quantile_encoding_' + target] = t1.mean(axis=1)

    return train_new, x_test


def third_quantile_encoding_single(x_train, x_test, col='work_type', target='stroke'):
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=123)
    t1 = pd.DataFrame()
    for i, (train_index, val_index) in enumerate(kf.split(x_train)):
        x_tr, x_val = x_train.iloc[train_index], x_train.iloc[val_index]
        p_class_encoding = x_val[col].map(x_tr.groupby([col])[target].quantile(.75))
        x_val[col + '_3quantile_encoding_' + target] = x_val[target] / p_class_encoding
        test_encoding = x_test[col].map(x_tr.groupby([col])[target].quantile(.75))
        t1['test_encoding' + '_' + str(i)] = test_encoding
        if i == 0:
            train_new = x_val
        else:
            train_new = pd.concat([train_new, x_val])
    x_test[col + '_3quantile_encoding_' + target] = t1.mean(axis=1)

    return train_new, x_test

##############Reading in the data#######
path = 'C:/Users/Parikshit/OneDrive/analyticswizard/'

train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
#train = pd.read_csv(path+'train1.csv')

num_cols = ['age','length_of_service','avg_training_score']

for i in num_cols:
    train[i] = train[i]*1.0
    test[i] = test[i]*1.0

train['previous_year_rating'].fillna(0,inplace=True)
test['previous_year_rating'].fillna(0,inplace=True)

train['previous_year_rating'] = train['previous_year_rating'].astype(int)
test['previous_year_rating'] = test['previous_year_rating'].astype(int)

print(train.dtypes)
print(test.dtypes)

#treating missing values

print(train.isnull().sum())
print(test.isnull().sum())

train['education'].fillna('unknown',inplace=True)
test['education'].fillna('unknown',inplace=True)


##train test split

from sklearn.model_selection import train_test_split

x = train.copy()
del x['is_promoted']
del x['employee_id']
y = train['is_promoted']

x_tr1, X_val, y_tr1, y_val = train_test_split(x,y, test_size=0.1, random_state=123,shuffle=True)
x_train, X_test, y_train, y_test = train_test_split(x_tr1,y_tr1, test_size=0.3, random_state=123,shuffle=True)

#######################################################baseline model#####################################################
from sklearn.metrics import *
import catboost as cb

model3= cb.CatBoostClassifier(iterations=1000,
                        learning_rate=0.05,
                        depth=7,
                        l2_leaf_reg=15,
                        rsm=None,
                        loss_function='CrossEntropy',
                        border_count=None,
                        feature_border_type='MinEntropy',
                        fold_permutation_block_size=None,
                        od_pval=None,
                        od_wait=None,
                        od_type=None,
                        nan_mode=None,
                        counter_calc_method=None,
                        leaf_estimation_iterations=None,
                        leaf_estimation_method=None,
                        thread_count=15,
                        random_seed=123,
                        use_best_model=True,
                        has_time=False,
                        one_hot_max_size=None,
                        random_strength=1,
                        name='experiment',
                        ignored_features=None,
                        train_dir=None,
                        custom_metric='Logloss',
                        eval_metric='Logloss',
                        bagging_temperature=0.8,
                        save_snapshot=None,
                        snapshot_file=None,
                        fold_len_multiplier=None,
                        #class_weights = [1,10.74],
                        #class_weights = [1,0.066769087],
                        # used_ram_limit=8,
                        # gpu_ram_part= 0.75,
                        allow_writing_files=None,
                        approx_on_full_history=None,
                        # task_type='GPU',
                        device_config=None,
                        verbose=False
                            )

categorical_features_indices = np.where(x_train.dtypes != np.float)[0]
bst = model3.fit(x_train, y_train,cat_features=categorical_features_indices, eval_set=(X_test, y_test))

pred_test = bst.predict_proba(X_test)[:,1]
fac = roc_auc_score(y_test, pred_test, average='macro', sample_weight=None)

pred_test_actual = np.where(pred_test>0.26,1,0)

#pred_test_actual = bst.predict(X_test)
f1 = f1_score(pred_test_actual,y_test,pos_label=1,average='binary')
print(f1)

#train_test = X_test.copy()
#train_test['actual'] = y_test
#train_test['pred'] = pred_test
#train_test.to_csv(path+'train_test_best.csv',index=False)

pred_val = bst.predict_proba(X_val)[:,1]
auc_val = roc_auc_score(y_val, pred_val, average='macro', sample_weight=None)

pred_val_actual = np.where(pred_val>0.291,1,0)

#pred_val_actual = bst.predict(X_val)
f1 = f1_score(pred_val_actual,y_val,pos_label=1,average='binary')
print(f1)

#train_val = x_val.copy()
#train_val['actual'] = y_val
#train_val['pred'] = pred_val
#train_val.to_csv(path+'train_val_best.csv',index=False)

importances = pd.Series(bst.feature_importances_, index=x_train.columns)
importances.sort_values(inplace=True, ascending=False)
importances.plot(kind='barh')


###submission##

test1 = test.copy()
del test1['employee_id']

predicted_test_prob = bst.predict_proba(test1)[:,1]
predicted_test = np.where(predicted_test_prob>0.296,1,0)

test['pred_prob'] = predicted_test_prob
test['is_promoted'] = predicted_test
submission = test[['employee_id','is_promoted']]

submission.to_csv(path+'submission_check1.csv',index=False)
###############################

#using only the important variables
imp = pd.DataFrame(importances).reset_index()
imp= imp.rename(columns={'index':'var',0:'importance'})
imp['cumulative'] = imp['importance'].cumsum()
imp = imp[imp['importance']>0]
cols_imp = list(imp['var'].unique())
top_20 = list(imp['var'].head(20))
