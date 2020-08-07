'''
Author: Junbong Jang
Date 4/30/2020

Store various models that forecast next day case/death numbers for one county
'''

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA

def baseline_model(train_X, train_y, test_X, test_y):
    predictions = list()
    for index, x in enumerate(test_X):
        if index == 0:
            yhat = train_y[-1]
        else:
            yhat = test_y[index-1]
        predictions.append(yhat)
    return predictions


def ARIMA_model(train_y, test_y):
    history = [x for x in train_y]
    predictions = list()
    for t in range(len(test_y)):
        model = ARIMA(history, order=(5, 0, 0))
        model_fit = model.fit(disp=0)
        # print(model_fit.summary())
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_y[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    return predictions

    # plot residual errors
    # residuals = pd.DataFrame(model_fit.resid)
    # residuals.plot()
    # plt.show()
    # residuals.plot(kind='kde')
    # plt.show()

def rf_model(train_y, test_y):
    history_days = [x for x in range(len(train_y))]
    history_val = [y for y in train_y]
    predictions = list()
    for t in range(len(test_y)):
        model = RandomForestRegressor(random_state=0)
        model.fit([history_days], [history_val])
        # print(model_fit.summary())
        next_day = history_days[-1]+1
        history_days.append(next_day)
        print(next_day)
        print(len(history_days[len(history_days)-67:]))
        output = model.predict([history_days[len(history_days)-67:]])
        yhat = output[0]
        predictions.append(yhat)
        obs = test_y[t]
        history_val.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    return predictions


def evaluate_predictions(test_y, predictions):
    error = mean_squared_error(test_y, predictions)
    print('Test MSE: %.3f' % error)
