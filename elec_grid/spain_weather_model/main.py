from elec_grid.spain_weather_model import sarima_train, xgb_train
import pandas as pd
import matplotlib.pyplot as plt

def get_outputs():
    df_pred_xgb = xgb_train.xgb_main()
    df_pred_sarima = sarima_train.sarima_main()

    df_preds = pd.DataFrame(zip(df_pred_sarima['actuals'], df_pred_sarima['predictions'],df_pred_xgb['temp_pred'].reset_index(drop=True)),columns = ['actuals','sarima','xgb'])
    df_preds = df_preds.iloc[-25:]
    plt.plot(df_preds,label = ['actual','sarima','xgb'])
    plt.legend()
