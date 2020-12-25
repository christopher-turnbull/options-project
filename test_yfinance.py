import yfinance as yf
import pandas as pd
# import matplotlib
# matplotlib.use('MacOSX')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA


google = yf.Ticker("GOOG")

#google.info returns json object

df = google.history(period='1d',
                    interval = "1m",)

low_df = df[['Low']]
low_df['date'] = pd.to_datetime(low_df.index).time
low_df.set_index('date',inplace = True)


X = df.index.values
y = df['Low'].values

offset = int(0.1*len(df))

X_train = X[:-offset]
y_train = y[:-offset]
X_test = X[-offset:]
y_test = X[-offset:]

plt.plot(range(0,len(y_train)),y_train,label='Train')
plt.show()


model = ARIMA(y_train,order=(5,0,1)).fit()

forecast = model.forecast(steps=1)[0]

print('Real data for time 0: %f' % y_train[len(y_train) - 1])
print('Real data for time 1: %f' % y_test[0])
print('pred data for time 1: %f' % forecast)
# this is terrible

