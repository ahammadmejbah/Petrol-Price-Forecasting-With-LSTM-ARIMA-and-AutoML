<div align="center">
      <h1> Petrol Price Forecasting With LSTM ,  ARIMA and AutoML</h1>
     </div>
<p align="center"> <a href="https://github.com/ahammadmejbah" target="_blank"><img alt="" src="https://img.shields.io/badge/Website-EA4C89?style=normal&logo=dribbble&logoColor=white" style="vertical-align:center" /></a> <a href="https://twitter.com/ahammadmejbah" target="_blank"><img alt="" src="https://img.shields.io/badge/Twitter-1DA1F2?style=normal&logo=twitter&logoColor=white" style="vertical-align:center" /></a> <a href="https://www.facebook.com/ahammadmejbah" target="_blank"><img alt="" src="https://img.shields.io/badge/Facebook-1877F2?style=normal&logo=facebook&logoColor=white" style="vertical-align:center" /></a> <a href="https://www.instagram.com/ahammadmejbah/" target="_blank"><img alt="" src="https://img.shields.io/badge/Instagram-E4405F?style=normal&logo=instagram&logoColor=white" style="vertical-align:center" /></a> <a href="https://www.linkedin.com/in/ahammadmejbah/}" target="_blank"><img alt="" src="https://img.shields.io/badge/LinkedIn-0077B5?style=normal&logo=linkedin&logoColor=white" style="vertical-align:center" /></a> </p>

ONGCF is an organization that works in the oil and natural gas industries. There is information accessible on the pricing on a weekly basis. It intends to make a forecast on the price of crude oil for the subsequent 16 months, beginning on January 1, 2019, and continuing through April 2020. Create a price forecast using the most effective model available according to your preferences. Check for any extreme values or values that are missing.

Train.training data and a submission example are both accessible in csv format. The format of the file that must be uploaded is specified as csv.

In this particular instance, we will be judging the quality of your work based on the MAPE assessment criteria. Within the first one hundred eighty minutes of the exam, you are required to hand in the submission. Posted applications that won't be considered for acceptance. A maximum of 20 contributions may be submitted.

    
### TimeLine of the project:
 1. Data Analysis
 2. Model Construction and Predictions Utilizing Machine Learning Techniques
 3. Model Construction and Prediction Employing Auto Keras (Auto ML)
 
 
 ### Importing Libraries
 
 ``` python
 import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
 
 ```

### Datasets Loading: 
``` Python
df1 = pd.read_csv("train_data.csv")
df1.head()
```

### Output: 

``` python
Date	Petrol (USD)
0	6/9/2003	74.59
1	6/16/2003	74.47
2	6/23/2003	74.42
3	6/30/2003	74.35
4	7/7/2003	74.28
```

### Datasets Pre-processing:

``` python

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)
```

### LSTM Architecture
The “long short-term memory” (LSTM) architecture is designed to meet the requirements of RNNs in need of a short-term memory that is capable of being retained for thousands of timesteps. A cell, an input gate, an output gate, and a forget gate are the component parts that make up a typical LSTM unit.


``` python
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

```

### Results of model performance

``` python
Epoch 1/10
7/7 [==============================] - 1s 172ms/step - loss: 0.1365 - val_loss: 0.0748
Epoch 2/10
7/7 [==============================] - 0s 32ms/step - loss: 0.0320 - val_loss: 0.0154
Epoch 3/10
7/7 [==============================] - 0s 29ms/step - loss: 0.0214 - val_loss: 0.0042
Epoch 4/10
7/7 [==============================] - 0s 26ms/step - loss: 0.0106 - val_loss: 0.0085
Epoch 5/10
7/7 [==============================] - 0s 27ms/step - loss: 0.0088 - val_loss: 0.0040
Epoch 6/10
7/7 [==============================] - 0s 28ms/step - loss: 0.0074 - val_loss: 0.0049
Epoch 7/10
7/7 [==============================] - 0s 28ms/step - loss: 0.0067 - val_loss: 0.0032
Epoch 8/10
7/7 [==============================] - 0s 27ms/step - loss: 0.0063 - val_loss: 0.0035
Epoch 9/10
7/7 [==============================] - 0s 27ms/step - loss: 0.0062 - val_loss: 0.0031
Epoch 10/10
7/7 [==============================] - 0s 32ms/step - loss: 0.0061 - val_loss: 0.0031
```

### Creating the model performance Visualiation

``` python
### Plotting
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df4)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df4)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df4)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df4))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
```

![Screenshot_1](https://user-images.githubusercontent.com/56669333/236427534-eba3ef04-f84d-4dd8-8c0c-dbaeb50cbf52.png)

### ARIMA
A statistical analysis model known as an autoregressive integrated moving average, or ARIMA, is one that makes use of time series data in order to either improve one’s understanding of the data set or to anticipate future trends. If a statistical model makes predictions about future values based on previous values, then the model is said to be autoregressive.

``` python
model1 = ARIMA(df4.values, order=(5,1,0))
model_fit1 = model1.fit()
output1= model_fit1.forecast(steps=30)
output1
```
### Output: 

``` python
array([119.80670597, 119.52143349, 119.3063469 , 119.14818594,
       119.0292548 , 118.93899202, 118.87071709, 118.81917614,
       118.78030433, 118.75097704, 118.72884006, 118.71212995,
       118.69951742, 118.68999816, 118.68281356, 118.67739093,
       118.67329814, 118.67020907, 118.66787756, 118.66611783,
       118.66478967, 118.66378722, 118.66303062, 118.66245956,
       118.66202856, 118.66170325, 118.66145772, 118.66127241,
       118.66113254, 118.66102697])
```

### AutoML
The process of automating the activities involved in applying machine learning to challenges that are found in the real world is referred to as automated machine learning. It is possible for AutoML to include every step, starting with the collection of raw data and ending with the construction of a machine learning model that is ready for deployment.

``` python
# evaluate the model
mae, _  = reg.evaluate(X_test, ytest, verbose=0)
#print('MAE: %.3f' % mae)
# use the model to make a prediction
yhat_test = reg.predict(X_test)

# get the best performing model
model = reg.export_model()
```
