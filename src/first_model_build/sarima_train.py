from statsmodels.tsa.arima.model import ARIMA
from data_prep import load_clean_weather_data
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
# build a sarima model for a feature using time series data only
def sarima_train(df_we, feature, n_predict):

    order = (2,0,2) # AR, one step DIFF, MA parts
    season_order = (1,1,0,24) # SEASONAL AR, diff, ma, lag parts
    lookback = 250 # number of rows to fit data on (converges at about this number)
    values= n_predict # number of hours/next vals to predict

    # create a model for each time step and predict
    predictions = []
    actuals = []
    print('Running models for each time step')
    for i in tqdm([int(x) for x in np.arange(values)]):
        df_train = pd.concat([df_we[df_we['sample']=='train'][feature].iloc[-lookback + i:-1], df_we[df_we['sample']=='test'][feature].iloc[:i]])
        df_test = df_we[df_we['sample']=='test'][feature].iloc[i:i+1]
        model = ARIMA(df_train, order = order, seasonal_order = season_order)
        model = model.fit()
        prediction = model.forecast(1).values[0]
        predictions.append(prediction)
        actuals.append(df_test.values[0])
    
    # SAVE PREDS AND ACTUALS
    df_actuals = df_we[df_we['sample']=='train'][feature].iloc[-lookback:].reset_index().rename(columns = {feature:'actuals'})
    df_predictions = pd.DataFrame(data = zip(predictions,actuals), columns = ['predictions','actuals'])
    df_predictions = pd.concat([df_actuals,df_predictions]).drop('index',axis=1).reset_index(drop=True)
    
    # get outputs
    print('rmse: ', root_mean_squared_error(df_predictions['actuals'].iloc[-values:],df_predictions['predictions'].iloc[-values:]))
    plt.plot(df_predictions['actuals'], label='actuals')
    plt.plot(df_predictions['predictions'], label='predictions')
    plt.legend()
    return df_predictions

def main(feature='temp',n_predict=25):
    df_we = load_clean_weather_data()
    df_predictions = sarima_train(df_we, feature = feature, n_predict = n_predict)
    return df_predictions

if __name__ == '__main__':
    main()
