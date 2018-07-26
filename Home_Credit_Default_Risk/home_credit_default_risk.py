import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def numerical_nan(df):
    numerical_columns = [col for col in df.columns if (((df[col].dtype == 'int64') or (df[col].dtype == 'float64')) and df[col].isna().sum() > 0)]
    for col in numerical_columns:
        df[col + '_NAN'] = df[col].isnull()
    return df
    
# Preprocess application_train.csv and application_test.csv
def application_df(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../datasets/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../datasets/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Xudong, drop NAME_CONTRACT_TYPE since the testing data only has 1%
    df = df.drop(['NAME_CONTRACT_TYPE'], axis =1)
    df['OWN_CAR_AGE'] = df['OWN_CAR_AGE'].fillna(0)
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['TOTAL_INQUIRIES'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] + df['AMT_REQ_CREDIT_BUREAU_DAY'] + df['AMT_REQ_CREDIT_BUREAU_WEEK'] + df['AMT_REQ_CREDIT_BUREAU_MON'] + df['AMT_REQ_CREDIT_BUREAU_QRT'] + df['AMT_REQ_CREDIT_BUREAU_YEAR']
    df['TOTAL_RECENT_INQUIRIES'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] + df['AMT_REQ_CREDIT_BUREAU_DAY'] + df['AMT_REQ_CREDIT_BUREAU_WEEK']
    df['RECENT_VS_TOTAL_INQUIRIES'] = df['TOTAL_RECENT_INQUIRIES'] / df['TOTAL_INQUIRIES']    
    
    df['NEW_EXT_SOURCES_MAX'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    df['NEW_EXT_SOURCES_MIN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    
    epslion = 1
    
    df['WEEKDAY_APPR_PROCESS_START'] = df['WEEKDAY_APPR_PROCESS_START'].replace({'MONDAY':0, 'TUESDAY':0,'WEDNESDAY':0,'THURSDAY':0,'FRIDAY':0, 'SATURDAY':1, 'SUNDAY':1})
    df['DIFF_ID_PUBLISH_REGISTRATION'] = df['DAYS_ID_PUBLISH'] - df['DAYS_REGISTRATION']
    df['DIFF_ID_PUBLISH_EMPLOYED'] = df['DAYS_ID_PUBLISH'] - df['DAYS_EMPLOYED']
    df['DIFF_ID_PUBLISH_BIRTH'] = df['DAYS_ID_PUBLISH'] - df['DAYS_BIRTH']
    df['DIFF_REGISTRATION_EMPLOYED'] = df['DAYS_REGISTRATION'] - df['DAYS_EMPLOYED']
    df['DIFF_EMPLOYED_BIRTH'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
    df['DIFF_PHONE_CHANGE_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_BIRTH']
    df['DIFF_PHONE_CHANGE_EMPLOYED'] = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_EMPLOYED']
    df['DIFF_PHONE_CHANGE_REGISTRATION'] = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_REGISTRATION']
    df['DIFF_PHONE_CHANGE_ID_PUBLISH'] = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_ID_PUBLISH']
    df['DIFF_PHONE_CHANGE_OWN_CAR'] = df['DAYS_LAST_PHONE_CHANGE']/365 - df['OWN_CAR_AGE']
    df['DIFF_OWN_CAR_AGE_DAYS_BIRTH'] = df['OWN_CAR_AGE'] - df['DAYS_BIRTH']/365
    df['DIFF_OWN_CAR_AGE_DAYS_EMPLOYED'] = df['OWN_CAR_AGE'] - df['DAYS_EMPLOYED']/365
    df['DIFF_OWN_CAR_AGE_REGISTRATION'] = df['OWN_CAR_AGE'] - df['DAYS_REGISTRATION']/365
    #df['PCT_OWN_CAR_AGE_DAYS_BIRTH'] = df['OWN_CAR_AGE'] / (df['DAYS_BIRTH'] - epslion)/365
    #df['PCT_OWN_CAR_AGE_DAYS_EMPLOYED'] = df['OWN_CAR_AGE'] / (df['DAYS_EMPLOYED'] - epslion)/365
    df['PCT_DAYS_EMPLOYED_BIRTH'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] - epslion)
    #df['PCT_ID_PUBLISH_REGISTRATION'] = df['DAYS_ID_PUBLISH'] / df['DAYS_REGISTRATION']
    #df['PCT_ID_PUBLISH_EMPLOYED'] = df['DAYS_ID_PUBLISH'] / df['DAYS_EMPLOYED']
    df['PCT_ID_PUBLISH_BIRTH'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['PCT_DAYS_REGISTRATION_BIRTH'] = df['DAYS_REGISTRATION'] / (df['DAYS_BIRTH'] - epslion)
    df['PCT_DAYS_REGISTRATION_EMPLOYED'] = df['DAYS_REGISTRATION'] / (df['DAYS_EMPLOYED'] - epslion)
    #df['PCT_PHONE_CHANGE_DAYS_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    #df['PCT_PHONE_CHANGE_DAYS_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / (df['DAYS_EMPLOYED'] - epslion)
    #df['PCT_PHONE_CHANGE_REGISTRATION'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_REGISTRATION']
    #df['PCT_PHONE_CHANGE_ID_PUBLISH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_ID_PUBLISH']
    df['PCT_INCOME_CREDIT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['PCT_INCOME_PRICE'] = df['AMT_INCOME_TOTAL'] / df['AMT_GOODS_PRICE'] 
    df['PCT_INCOME_FAM_MEMBERS'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['PCT_CREDIT_FAM_MEMBERS'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    df['PCT_ANNUITY_FAM_MEMBERS'] = df['AMT_ANNUITY'] / df['CNT_FAM_MEMBERS']
    #df['PCT_INCOME_CNT_CHILDREN'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + epslion)
    #df['PCT_ANNUITY_INCOME_CNT_CHILDREN'] = df['AMT_ANNUITY'] / df['PCT_INCOME_CNT_CHILDREN']
    df['PCT_ANNUITY_INCOME_FAM_MEMBERS'] =  df['AMT_ANNUITY'] / df['PCT_INCOME_FAM_MEMBERS']
    df['PCT_ANNUITY_PRICE'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE'] 
    #df['PCT_CHILDREN_FAM_MEMBER'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['STD_EXT_SOURCE'] =  df[[x for x in df.columns if x.startswith('EXT_SOURCE')]].std(axis=1)
    df['CNT_NULL_EXT_SOURCE'] = df[[x for x in df.columns if x.startswith('EXT_SOURCE')]].isnull().sum(axis=1)
    #df['GMEAN_EXT_SOURCE'] = gmean(df[[x for x in df.columns if x.startswith('EXT_SOURCE')]], axis=1)
    df['CNT_AMT_REQ_CREDIT_BUREAU'] = df[[x for x in df.columns if x.startswith('AMT_REQ_CREDIT_BUREAU')]].sum(axis=1)
    df['SUM_OBS_CNT_SOCIAL_CIRCLE'] = df[['DEF_30_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']].sum(axis=1)
    df['ORGANIZATION_TYPE'] = df.ORGANIZATION_TYPE.apply(lambda x: x.split()[0])
    df = df.merge(df[['OCCUPATION_TYPE', 'AMT_INCOME_TOTAL']].groupby('OCCUPATION_TYPE').mean().rename({'AMT_INCOME_TOTAL':'MEAN_INCOME_OCCUPATION'},axis=1), left_on='OCCUPATION_TYPE', right_index=True, how='left')
    df = df.merge(df[['OCCUPATION_TYPE', 'AMT_CREDIT']].groupby('OCCUPATION_TYPE').mean().rename({'AMT_CREDIT':'MEAN_CREDIT_OCCUPATION'},axis=1), left_on='OCCUPATION_TYPE', right_index=True, how='left')
    df = df.merge(df[['OCCUPATION_TYPE', 'AMT_ANNUITY']].groupby('OCCUPATION_TYPE').mean().rename({'AMT_ANNUITY':'MEAN_ANNUITY_OCCUPATION'},axis=1), left_on='OCCUPATION_TYPE', right_index=True, how='left')
    df = df.merge(df[['ORGANIZATION_TYPE', 'AMT_INCOME_TOTAL']].groupby('ORGANIZATION_TYPE').mean().rename({'AMT_INCOME_TOTAL':'MEAN_INCOME_ORGANIZATION'},axis=1), left_on='ORGANIZATION_TYPE', right_index=True, how='left')
    df = df.merge(df[['ORGANIZATION_TYPE', 'AMT_CREDIT']].groupby('ORGANIZATION_TYPE').mean().rename({'AMT_CREDIT':'MEAN_CREDIT_ORGANIZATION'},axis=1), left_on='ORGANIZATION_TYPE', right_index=True, how='left')
    df = df.merge(df[['ORGANIZATION_TYPE', 'AMT_ANNUITY']].groupby('ORGANIZATION_TYPE').mean().rename({'AMT_ANNUITY':'MEAN_ANNUITY_ORGANIZATION'},axis=1), left_on='ORGANIZATION_TYPE', right_index=True, how='left')
    df['MEAN_INCOME_OCCUPATION'] = (df['AMT_INCOME_TOTAL'] - df['MEAN_INCOME_OCCUPATION']) / df['MEAN_INCOME_OCCUPATION']
    df['MEAN_CREDIT_OCCUPATION'] = (df['AMT_CREDIT'] - df['MEAN_CREDIT_OCCUPATION']) / df['MEAN_CREDIT_OCCUPATION']
    df['MEAN_ANNUITY_OCCUPATION'] = (df['AMT_ANNUITY'] - df['MEAN_ANNUITY_OCCUPATION']) / df['MEAN_ANNUITY_OCCUPATION']
    df['MEAN_INCOME_ORGANIZATION'] = (df['AMT_INCOME_TOTAL'] - df['MEAN_INCOME_ORGANIZATION']) / df['MEAN_INCOME_ORGANIZATION']
    df['MEAN_CREDIT_ORGANIZATION'] = (df['AMT_CREDIT'] - df['MEAN_CREDIT_ORGANIZATION']) / df['MEAN_CREDIT_ORGANIZATION']
    df['MEAN_ANNUITY_ORGANIZATION'] = (df['AMT_ANNUITY'] - df['MEAN_ANNUITY_ORGANIZATION']) / df['MEAN_ANNUITY_ORGANIZATION']

    
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    del test_df
    gc.collect()
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../datasets/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../datasets/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
        
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'min', 'max', 'mean', 'var' , 'sum'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean', 'sum'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_MAX_OVERDUE': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_SUM': [ 'min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['min', 'max', 'mean', 'sum'],
        'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['min', 'max', 'mean', 'sum'],
        'MONTHS_BALANCE_MIN': ['min', 'max', 'mean', 'sum'],
        'MONTHS_BALANCE_MAX': ['min', 'max', 'mean', 'sum'],
        'MONTHS_BALANCE_SIZE': ['min', 'max', 'mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    
    #active_agg['SK_ID_CURR'] = active_agg.index
    #bureau_agg['SK_ID_CURR'] = bureau_agg.index
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    #bureau_agg = pd.merge(bureau_agg, active_agg, how='left', on='SK_ID_CURR')
    
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
 
    # Counts number of loans for each customer
    bureau_agg['LOAN_COUNTS'] = bureau.groupby(['SK_ID_CURR']).size()
    
    bureau_agg['LOAN_COUNTS_PER_YEAR'] = bureau_agg['LOAN_COUNTS'] / (bureau_agg['BURO_DAYS_CREDIT_MEAN'] /365)
    bureau_agg['DEBT_USAGE'] = bureau_agg['BURO_AMT_CREDIT_SUM_DEBT_MEAN'] / (bureau_agg['BURO_AMT_CREDIT_SUM_LIMIT_MEAN'])
    bureau_agg['OVERDUE_PERCENT_MEAN'] = bureau_agg['BURO_AMT_CREDIT_SUM_OVERDUE_MEAN'] / (bureau_agg['BURO_AMT_CREDIT_SUM_DEBT_MEAN'])
    bureau_agg['OVERDUE_PERCENT_MAX'] = bureau_agg['BURO_AMT_CREDIT_SUM_OVERDUE_MAX'] / (bureau_agg['BURO_AMT_CREDIT_SUM_DEBT_MAX'])
    bureau_agg['OVERDUE_PERCENT_SUM'] = bureau_agg['BURO_AMT_CREDIT_SUM_OVERDUE_SUM'] / (bureau_agg['BURO_AMT_CREDIT_SUM_DEBT_SUM'])
    
    bureau_agg['DEBT_USAGE'].replace(np.inf, np.nan, inplace=True)
    bureau_agg['OVERDUE_PERCENT_MEAN'].replace(np.inf, np.nan, inplace=True)
    bureau_agg['OVERDUE_PERCENT_MAX'].replace(np.inf, np.nan, inplace=True)
    bureau_agg['OVERDUE_PERCENT_SUM'].replace(np.inf, np.nan, inplace=True)
    
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../datasets/previous_application.csv', nrows = num_rows)
    
    # Xudong, drop 100% NaN
    prev = prev.drop(['RATE_INTEREST_PRIMARY'], axis =1)
    prev = prev.drop(['RATE_INTEREST_PRIVILEGED'], axis = 1)
    
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['DOWN_PAYMENT_TO_PRICE'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_GOODS_PRICE']
    
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'sum', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean', 'sum'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_DECISION': ['min', 'max', 'mean', 'sum'],
        'CNT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DOWN_PAYMENT_TO_PRICE': ['min', 'max', 'mean', 'sum']
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../datasets/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
        'SK_DPD': ['min', 'max', 'mean', 'sum'],
        'SK_DPD_DEF': ['min', 'max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../datasets/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['min', 'max', 'mean', 'sum'],
        'DBD': ['min', 'max', 'mean', 'sum'],
        'PAYMENT_PERC': ['min', 'max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../datasets/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)

    # add new features 
    cc['PREV_BALANCE_TO_LIMIT'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['CURR_BALANCE_TO_LIMIT'] = cc['MONTHS_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['BALANCE_DIFF'] = cc['CURR_BALANCE_TO_LIMIT'] - cc['PREV_BALANCE_TO_LIMIT']
    cc['DRAWINGS_TO_BALANCE'] = cc['AMT_DRAWINGS_CURRENT'] / cc['AMT_BALANCE']
    cc['ATM_DRAWINGS_TO_BALANCE'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_BALANCE']
    cc['DRAWSINGS_TO_LIMIT'] = cc['AMT_DRAWINGS_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['CURR_PAYMENT_TO_BALANCE'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_BALANCE']
    cc['CURR_PAYMENT_TO_LIMIT'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['TOTAL_PAYMENT_TO_BALANCE'] = cc['AMT_PAYMENT_TOTAL_CURRENT'] / cc['AMT_BALANCE']
    cc['TOTAL_PAYMENT_TO_LIMIT'] = cc['AMT_PAYMENT_TOTAL_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    cc['CURR_PAYMENT_TO_TOTAL_PAYMENT'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_PAYMENT_TOTAL_CURRENT']
    
    cc['AVG_ATM_DRAWINGS'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['CNT_DRAWINGS_ATM_CURRENT']
    cc['AVG_DRAWINGS'] = cc['AMT_DRAWINGS_CURRENT'] / cc['CNT_DRAWINGS_CURRENT']
    cc['AVG_DRAWINGS_OTHER'] = cc['AMT_DRAWINGS_OTHER_CURRENT'] / cc['CNT_DRAWINGS_OTHER_CURRENT']
    cc['AVG_DRAWINGS_POS'] = cc['AMT_DRAWINGS_POS_CURRENT'] / cc['CNT_DRAWINGS_POS_CURRENT']
    
    cc['RECEIVABLE_TO_BALANCE'] = cc['AMT_RECIVABLE'] / cc['AMT_BALANCE']
    cc['RECEIVABLE_PRINCIPAL_TO_BALANCE'] = cc['AMT_RECEIVABLE_PRINCIPAL'] / cc['AMT_BALANCE']
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    
    del cc
    gc.collect()
    return cc_agg

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = True, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=48,
            #is_unbalance=True,
            n_estimators=10000,
            learning_rate=0.02, # 0.02
            num_leaves=32,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.04,
            reg_lambda=0.073,
            min_split_gain=0.0222415,
            min_child_weight=40,
            silent=-1,
            verbose=-1,
            #scale_pos_weight=11
            )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df

# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_df(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        
        df['AMT_CREDIT_SUM_TO_INCOME'] = bureau['BURO_AMT_CREDIT_SUM_MEAN'] / df['AMT_INCOME_TOTAL']
        df['AMT_CREDIT_SUM_DEBT_TO_INCOME'] = bureau['BURO_AMT_CREDIT_SUM_DEBT_MEAN'] / df['AMT_INCOME_TOTAL']
        df['INCREASE_AMT_CREDIT_PERC'] = df['AMT_CREDIT'] / bureau['BURO_AMT_CREDIT_SUM_SUM']
        df['NEW_CREDIT_TO_ACTIVE_CREDIT_SUM'] = df['AMT_CREDIT'] / bureau['ACTIVE_AMT_CREDIT_SUM_SUM']
        df['DAYS_CREDIT_TO_BIRTHDAY'] = bureau['BURO_DAYS_CREDIT_MEAN'] / df['DAYS_BIRTH']
        df['DAYS_CREDIT_TO_DAYS_EMPLOYED'] = bureau['BURO_DAYS_CREDIT_MEAN'] / df['DAYS_EMPLOYED']
        
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        del cc
        gc.collect()
        
        # df = numerical_nan(df)
    with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5, stratified= False, debug= debug)

if __name__ == "__main__":
    submission_file_name = "submission_curr.csv"
    with timer("Full model run"):
        main()
        
       
