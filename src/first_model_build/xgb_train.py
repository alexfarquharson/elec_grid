from statsmodels.tsa.arima.model import ARIMA
from data_prep import load_clean_weather_data
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor, XGBClassifier
import warnings
warnings.filterwarnings("ignore")


def get_prev_values(df, feature, lag_values_list):
    # get prev 1 hr,2hr,24hr,last year - could also add last year mean
    for x in lag_values_list:
        df[f'{feature}_{str(x)}'] = df[f'{feature}'].shift(x)
    return df

def create_first_df(df, features, lag_values_list):
    # create first df as the time features
    # fill the feature_pred with the previous -1 value
    for feature in features:
        df = get_prev_values(df, feature, lag_values_list)
        df[f'{feature}_pred'] = df[f'{feature}_1']
    df = df[df['sample'].isin(['train','test'])]
    return df

def fit_predict(df, target, model_features, reg_class):
    params = {'colsample_bytree':0.8, 'max_depth' : 4}
    if reg_class == 'reg':
        model = XGBRegressor(**params)
    else:
        model = XGBClassifier(**params)
    df_train = df[df['sample']=='train']
    model.fit(df_train[model_features], df_train[target])
    df[f'{target}_pred_challenger'] = model.predict(df[model_features])
    return df, model



def update_feature_pred_values(df, feature, convergence_dict, iteration):

    # if this score is best score then update pred values
    if convergence_dict[feature]['scores'][-1] == convergence_dict[feature]['scores_best']:
        df[f'{feature}_pred'] = df[f'{feature}_pred_challenger']
        print(f'updated preds values, {feature}, {iteration}')
    return df

def xgb_evaluate(df, features, convergence_dict):

    # to match sarima evaluation get same time frame
    lookback = 250
    n_predict = 25
    indices = df[df['sample']=='train'].index[-lookback:].append(df[df['sample']=='test'].index[:n_predict])
    df = df[df.index.isin(indices)]
    
    for feature in features:
        print('feature importances: ', convergence_dict[feature]['model'].feature_importances_)
        # get rmse of test
        rmse = root_mean_squared_error(df.iloc[-25:][feature], df.iloc[-25:][f'{feature}_pred'])
        print('rmse: ', rmse)

        plt.figure(figsize=(8, 4))  # new figure for each feature
        plt.plot(df.reset_index()[feature],label ='true')
        plt.plot(df.reset_index()[f'{feature}_pred'],label ='pred')
        lst = [np.nan]*(lookback+n_predict)
        lst[lookback] = df.reset_index()[feature][lookback]
        plt.scatter(x = np.arange((lookback+n_predict)), y = lst, c='r')
        plt.legend()
        plt.title(f"True vs Predicted - {feature}", fontsize=14)
    return df

def xgb_train(df_we, features, lag_values, model_iterations = 10):

    reg_class_dict = {'temp':'reg','humidity':'reg','wind_speed':'reg','clear':'class','clouds':'class','rain':'class'}
    convergence_dict = {f: {'model':np.nan,'scores' : [], 'scores_min':np.nan} for f in features}
    # create first df -  train model on lag features and previous other weather features
    df_model = create_first_df(df_we, features, lag_values)
    # for each feature build a model - then rebuild the model using the predicted other features - i fmodel improvement then overwrite model otherwise keep old model
    for it in np.arange(model_iterations):
        for feature in features:   
            # build and get predictions on model built on time features for var and predicted feature for other features
            # fill in to get best guess for pred feature
            model_time_features = [f'{feature}_{int(x)}' for x in lag_values]
            model_other_features = [f'{f}_pred' for f in features if f != feature]
            model_features = model_time_features + model_other_features
            df_model, model = fit_predict(df_model, target = feature, model_features = model_features, reg_class = reg_class_dict[feature])
            convergence_dict[feature]['model'] = model
            convergence_dict[feature]['scores'].append(root_mean_squared_error(df_model[df_model['sample'] =='test'][feature], df_model[df_model['sample'] =='test'][f'{feature}_pred_challenger']))
            convergence_dict[feature]['scores_best'] = min(convergence_dict[feature]['scores'])
            # todo - the model is diverging instead of converging with future iterations (as we update the x_pred of the other features)
            # simple fix - avoid this by only updating the preds if this leads to a better prediction
            df_model = update_feature_pred_values(df_model, feature, convergence_dict, it)

    # evaluate
    df_predictions = xgb_evaluate(df_model, features, convergence_dict)

    return df_predictions

def main(features = ['temp'], model_iterations = 1):
    df_we = load_clean_weather_data()
    # train xgb model on lag features and other weather features
    # diff with sarima is just train on emodel on train and predict on test (sarima) trains an individual model to predict one time step ahead
    lag_values = [int(x) for x in np.arange(1,27)] #+ [8760, 8761]
    # features: ,'humidity','wind_speed','clear','clouds','rain']
    df_predictions = xgb_train(df_we, features, lag_values, model_iterations = model_iterations)
    return df_predictions

if __name__ == '__main__':
    main()
