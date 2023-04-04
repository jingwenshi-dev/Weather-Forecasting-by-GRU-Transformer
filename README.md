# Weather Forecast Model in Great Toronto Area by using GRU 

## Introduction

The objective of this project is to develop a Gated Recurrent Unit (GRU) model to predict the following 24-hour temperature with selected hourly meteorological data obtained from Weather Canada in the Greater Toronto Area. Precise weather forecasts are crucial for a variety of industries, such as agriculture, aviation, transportation, and public safety. The model will utilize continuous hourly weather data spanning a week as input and generate a prediction for the temperature over the ensuing 24-hour period as output.

GRU is a variation of the Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) model. In comparison to RNN, GRU mitigates the issues of gradient explosion and vanishing, while being generally consider as more computationly efficient than LSTM, without effecting the performance too much. Therefore, GRU has been chosen for this project instead of LSTM.



## Model

### Model Visualization

As a variation of RNN, when a sequence of time series data is fed into the model, GRU uses the previous hidden state and the new input to predict a new output and new hidden state. The update gate decides how much past information should be passed to the next iteration while the reset gate determines how much past information should be forgotten. 

### Parameters



### Examples



## Data

The data comes from [Weather Canada](https://climate.weather.gc.ca/climate_data/hourly_data_e.html?hlyRange=2009-12-10%7C2023-04-02&dlyRange=2010-02-02%7C2023-04-02&mlyRange=%7C&StationID=48549&Prov=ON&urlExtension=_e.html&searchType=stnName&optLimit=yearRange&StartYear=2015&EndYear=2022&selRowPerPage=25&Line=0&searchMethod=contains&Month=4&Day=2&txtStationName=Toronto+City+Centre&timeframe=1&Year=2023) and is observied by Toronto City Centre Weather Observatory and contains 2015-2022's data.



Since some of the data is missing, the missing value will be filled with the average of the two hours before and after.

The training set data contains all the weather data from 2015-2019. The validation set data used data comes from 2020 and 2021. The test set data is the 2022's data.
