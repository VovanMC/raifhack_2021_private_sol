import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.metrics import mean_squared_error
import math
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
import re
from scipy.optimize import minimize
from tqdm import tqdm
import pickle

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.metrics import mean_squared_error
import math
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
import re
from scipy.optimize import minimize
from tqdm import tqdm

def standart_split(data, ltr):
    split_list = []
    train_data = data.loc[:ltr-1, :]
    train_data = train_data[train_data.price_type == 1]
    kf = KFold(n_splits = 5, shuffle = True, random_state = 228)
    for train_index, test_index in kf.split(train_data):
        split_list += [(train_data.index[train_index], train_data.index[test_index])]
    return split_list

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

data = pd.concat([train, test], axis = 0).reset_index(drop = True)
ltr = len(train)

subm = pd.read_csv('test_submission.csv')

data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

clean_floor_dict = {}
for x in data[data.price_type == 1]['floor'].unique():
    if x != x:
        clean_floor_dict[x] = float(x)
    elif len(x) <= 4 and ',' not in x and '-' not in x[1:]:
        clean_floor_dict[x] = float(x)
    elif 'этаж' in x and len(x) < 7:
        clean_floor_dict[x] = float(x.replace('этаж', ' ').strip())
    elif x.lower() == 'подвал':
        clean_floor_dict[x] = -1
    elif x.lower() == 'цоколь':
        clean_floor_dict[x] = -1
    elif x[1:] == '-й':
        clean_floor_dict[x] = float(x[0])
    else:
        clean_floor_dict[x] = -10

data['floor'] = data['floor'].map(clean_floor_dict)
        
center_pos = {}
for city, lat, lng, dist in tqdm(data[['city', 'lat', 'lng', 'osm_city_closest_dist']].values):
  if center_pos.get(city) is None:
    center_pos[city] = (dist, lat, lng)
  elif center_pos[city][0] > dist:
    center_pos[city] = (dist, lat, lng)
data['dx'] = [lat - center_pos[city][1] if city in center_pos else np.nan for lat, city in data[['lat', 'city']].values ]
data['dy'] = [lng - center_pos[city][2] if city in center_pos else np.nan for lng, city in data[['lng', 'city']].values ]

for i in range(4):
    data[f'lat_{i}'] = data['lat'].apply(lambda x: round(x, i)).astype(str)
    data[f'lng_{i}'] = data['lng'].apply(lambda x: round(x, i)).astype(str)
    data[f'coord_{i}'] = data[f'lat_{i}'] + '_' + data[f'lng_{i}']

data['street_city'] = data['street'] + '_' + data['city']
data['city_floor'] = data['city'] + '_' + data['floor'].astype('str')


cat_cols = ['region', 'street', 'city', 'street_city', 'osm_city_nearest_name', 'coord_0', 'coord_1', 'coord_2']


my_stats = data.loc[:ltr-1, :].reset_index(drop = True)
my_stats = my_stats[my_stats.price_type == 0].reset_index(drop = True)  
for col in cat_cols:
  data['zeros_mean_target_stats_' + col] = data[col].map(my_stats.groupby(col)['per_square_meter_price'].mean())
  data['zeros_median_target_stats_' + col] = data[col].map(my_stats.groupby(col)['per_square_meter_price'].median())
  #data['min_target_stats_' + col] = data[col].map(my_stats.groupby(col)['per_square_meter_price'].min())
  #data['max_target_stats_' + col] = data[col].map(my_stats.groupby(col)['per_square_meter_price'].max())
#     data['enc_' + col] = data[col].map({x:i for i, x in enumerate(data[col].unique())})

THRESHOLD = 0.15
NEGATIVE_WEIGHT = 1.1

def deviation_metric_one_sample(y_true, y_pred):
    """
    Реализация кастомной метрики для хакатона.

    :param y_true: float, реальная цена
    :param y_pred: float, предсказанная цена
    :return: float, значение метрики
    """
    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= THRESHOLD:
        return 0
    elif deviation <= - 4 * THRESHOLD:
        return 9 * NEGATIVE_WEIGHT
    elif deviation < -THRESHOLD:
        return NEGATIVE_WEIGHT * ((deviation / THRESHOLD) + 1) ** 2
    elif deviation < 4 * THRESHOLD:
        return ((deviation / THRESHOLD) - 1) ** 2
    else:
        return 9


def deviation_metric(y_true, y_pred):
    return np.array([deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()

def lgb_deviation_metric_one_sample(y_true, y_pred):
    """
    Реализация кастомной метрики для хакатона.

    :param y_true: float, реальная цена
    :param y_pred: float, предсказанная цена
    :return: float, значение метрики
    """

    THRESHOLD = 0.15
    DELTA = 0.01
    NEGATIVE_WEIGHT = 1.0

    deviation = (y_pred - y_true) / np.maximum(1e-8, y_true)
    if np.abs(deviation) <= THRESHOLD - DELTA:
        return 0
    elif deviation <= - 4 * (THRESHOLD - DELTA):
        return 9 * NEGATIVE_WEIGHT
    elif deviation < -THRESHOLD:
        return NEGATIVE_WEIGHT * ((deviation / (THRESHOLD - DELTA)) + 1) ** 2
    elif deviation < 4 * THRESHOLD:
        return ((deviation / (THRESHOLD - DELTA)) - 1) ** 2
    else:
        return 9

def lbg_metric(y_true, y_pred):
    return np.array([lgb_deviation_metric_one_sample(y_true[n], y_pred[n]) for n in range(len(y_true))]).mean()

def feval_raif(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RAIF_SCORE', deviation_metric(y_true, y_pred), False

# def standart_split(data, ltr):
#     train_data = data.loc[:ltr-1, :]
#     split_list = []
#     for date in ['2020-06-15', '2020-07-01', '2020-07-15', '2020-08-01', '2020-08-15', '2020-09-01']:
#         valid_index = train_data[(train_data['date'] >= pd.to_datetime(date)) & (train_data['price_type'] == 1)].index
#         train_index = train_data[(train_data['date'] < pd.to_datetime(date))  & (train_data['price_type'] == 1)].index
#         split_list += [(train_index, valid_index)]
#     return split_list




def lgb_train(data, target, ltr, split_list, train_cols, param, v_e = 0, n_e = 10000, cat_col = None):
    global best_score
    pred = pd.DataFrame()
    pred_val = np.zeros(ltr)
    fi = np.zeros(data.shape[1])
    score = []
    bst_list = []
    pred_train = pd.DataFrame()
    j = 0
    target = target
    for i , (train_index, test_index) in enumerate(split_list):
        y_train, y_val = np.array(target)[train_index], np.array(target)[test_index]
        train_weights = 1 / y_train
        val_weights = 1 / y_val
        tr = lgb.Dataset(data.loc[train_index, :], y_train, weight = train_weights)
        te = lgb.Dataset(data.loc[test_index, :], y_val, weight = val_weights, reference=tr)
        evallist = [(tr, 'train'), (te, 'test')]
        if len(test_index) > 0:
            bst = lgb.train(param, tr, num_boost_round=n_e, 
                    valid_sets = [tr, te], early_stopping_rounds=int(5 / param['learning_rate']), feval = feval_raif, 
                            verbose_eval = v_e)
        else:
            print(1)
            bst = lgb.train(param, tr, num_boost_round=400, 
                    )            
#         return bst
        pred[str(i)] = bst.predict(np.array(data)[ltr:, :])
        pred_val[test_index] = bst.predict(np.array(data)[test_index]) 
        cur_score = deviation_metric(np.array(target)[test_index], pred_val[test_index])
        score += [cur_score]
        bst_list += [bst]
        fi += np.array(bst.feature_importance(importance_type = 'gain'))
        print(i+1, np.mean(score))
        print()
        del tr, te        

        global best_score
        if np.mean(score) > best_score:          
          #break
          pass


    pred_train[str(j)] = pred_val
    return bst_list, pred_train, pred, score, fi

param_lgb = {
    'bagging_fraction': 0.9, 
    'bagging_freq': 1, 
    'boost': 'gbdt', 
    'feature_fraction': 0.9, 
    'max_depth':6,
    'learning_rate': 0.05, 
    'metric': 'custom', 
    'objective': 'tweedie', 
#     'tweedie_variance_power':1.5,
#     'max_bin':200,
#     'lambda_l1':0.001,
    'reg_sqrt':True,
    'verbose': -1,
}

split_list = standart_split(data, ltr)

drop_cols = ['city', 'date', 'id', 'osm_city_nearest_name', 'per_square_meter_price', 'region', 'street', 
             'coord_0', 'coord_1', 'coord_2', 'coord_3', 'lat_0', 'month','day', 'tmp_x', 'tmp_y',
 'lng_0',
 'lat_1',
 'lng_1',
 'lat_2',
 'lng_2',
 'lat_3',
 'lng_3', 'street_city', 'coord_1_floor', 'city_floor']
train_cols = [x for x in data.columns if x not in drop_cols]
print(len(train_cols))
print(train_cols)

features = ['floor', 'lat', 'lng', 'osm_amenity_points_in_0.005', 'osm_amenity_points_in_0.0075', 'osm_building_points_in_0.005', 'osm_building_points_in_0.01', 'osm_catering_points_in_0.001', 'osm_catering_points_in_0.005', 'osm_catering_points_in_0.0075', 'osm_catering_points_in_0.01', 'osm_city_closest_dist', 'osm_city_nearest_population', 'osm_crossing_closest_dist', 'osm_crossing_points_in_0.001', 'osm_crossing_points_in_0.005', 'osm_crossing_points_in_0.01', 'osm_culture_points_in_0.001', 'osm_culture_points_in_0.005', 'osm_culture_points_in_0.0075', 'osm_culture_points_in_0.01', 'osm_finance_points_in_0.001', 'osm_finance_points_in_0.005', 'osm_finance_points_in_0.0075', 'osm_finance_points_in_0.01', 'osm_healthcare_points_in_0.005', 'osm_healthcare_points_in_0.0075', 'osm_healthcare_points_in_0.01', 'osm_historic_points_in_0.005', 'osm_historic_points_in_0.0075', 'osm_historic_points_in_0.01', 'osm_hotels_points_in_0.005', 'osm_hotels_points_in_0.0075', 'osm_hotels_points_in_0.01', 'osm_leisure_points_in_0.005', 'osm_leisure_points_in_0.0075', 'osm_leisure_points_in_0.01', 'osm_offices_points_in_0.001', 'osm_offices_points_in_0.005', 'osm_offices_points_in_0.0075', 'osm_offices_points_in_0.01', 'osm_shops_points_in_0.001', 'osm_shops_points_in_0.005', 'osm_shops_points_in_0.0075', 'osm_shops_points_in_0.01', 'osm_subway_closest_dist', 'osm_train_stop_closest_dist', 'osm_train_stop_points_in_0.005', 'osm_train_stop_points_in_0.0075', 'osm_train_stop_points_in_0.01', 'osm_transport_stop_closest_dist', 'osm_transport_stop_points_in_0.005', 'osm_transport_stop_points_in_0.0075', 'osm_transport_stop_points_in_0.01', 'reform_count_of_houses_1000', 'reform_count_of_houses_500', 'reform_house_population_1000', 'reform_house_population_500', 'reform_mean_floor_count_1000', 'reform_mean_floor_count_500', 'reform_mean_year_building_1000', 'reform_mean_year_building_500', 'total_square', 'realty_type', 'dx', 'dy', 'zeros_mean_target_stats_region', 'zeros_median_target_stats_region', 'zeros_mean_target_stats_street', 'zeros_median_target_stats_street', 'zeros_mean_target_stats_city', 'zeros_median_target_stats_city', 'zeros_mean_target_stats_street_city', 'zeros_median_target_stats_street_city', 'zeros_mean_target_stats_osm_city_nearest_name', 'zeros_median_target_stats_osm_city_nearest_name', 'zeros_mean_target_stats_coord_0', 'zeros_median_target_stats_coord_0', 'zeros_mean_target_stats_coord_1', 'zeros_median_target_stats_coord_1', 'zeros_mean_target_stats_coord_2', 'zeros_median_target_stats_coord_2']

bst_list, pred_train, pred, score, fi = lgb_train(data[features], data['per_square_meter_price'], ltr, split_list, features, param_lgb, 100)
data['pred'] = pred_train
list_df = [data.loc[x[1]] for x in split_list]
full_train = pd.concat(list_df).reset_index(drop=True)
data = data.drop(columns=['pred'])
cur_score = deviation_metric(full_train['per_square_meter_price'].values, full_train['pred'].values)

print(f"cur_score = {cur_score}")
#full_train.to_csv('train_with_predict.csv', index=False)

feat_cols = [x[1] for x in sorted(zip(fi, train_cols))][::-1]

data['pred'] = pred_train
list_df = [data.loc[x[1]] for x in split_list]
full_train = pd.concat(list_df).reset_index(drop=True)
data = data.drop(columns=['pred'])
print(f"current_error = {np.round(deviation_metric(full_train['per_square_meter_price'].values, full_train['pred'].values), 5)}")
full_train.to_csv('train_with_predict.csv', index=False)

train_targets = full_train['per_square_meter_price'].values
train_predicts = full_train['pred'].values
def minimize_arit(W):
    ypred = W[0] * train_predicts
    result = deviation_metric(train_targets, ypred)    
    return result

W = minimize(minimize_arit, [1.05], options={'gtol': 1e-6, 'disp': True}).x
print('Weights arit:', W)
print('DivBy', 1.0 / W[0])

train_targets = full_train['per_square_meter_price'].values
train_predicts = full_train['pred'].values
def minimize_arit_2(W):
    ypred = W[0] * train_predicts + W[1]
    result = deviation_metric(train_targets, ypred)    
    return result

W = minimize(minimize_arit_2, [1.05, 100000], options={'gtol': 1e-6, 'disp': True}).x
print('Weights arit:', W)
print('DivBy', 1.0 / W[0])

data.loc[ltr:, 'per_square_meter_price'] = pred.mean(axis = 1).values * W[0]

ans = data.loc[ltr:, ['id', 'per_square_meter_price']].reset_index(drop = True)

ans.to_csv('submission.csv', index = None)
