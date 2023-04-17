# Weather Forecast Model in Great Toronto Area by using GRU

## Introduction

The objective of this project is to develop a Gated Recurrent Unit (GRU) model to predict the following 24-hour temperature with selected hourly meteorological data obtained from Weather Canada in the Greater Toronto Area. Precise weather forecasts are crucial for a variety of industries, such as agriculture, aviation, transportation, and public safety. The model will utilize continuous hourly weather data spanning a week as input and generate a prediction for the temperature over the ensuing 24-hour period as output.

GRU is a variation of the Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM) model. In comparison to RNN, GRU mitigates the issues of gradient explosion and vanishing, while being generally consider as more computationly efficient than LSTM, without effecting the performance too much. Therefore, GRU has been chosen for this project instead of LSTM.

# Model

### Model Visualization

As a variation of RNN, when a sequence of time series data is fed into the model, GRU uses the previous hidden state and the new input to predict a new output and new hidden state. The update gate decides how much past information should be passed to the next iteration while the reset gate determines how much past information should be forgotten. 

### Parameters

-   Input size (input_size)
    -   The number of features for each piece of the data. In this project, it will be 6.
        -   i.e. Temperature (Temp), Dew Point Temp (°C), Relative Humidity (Rel Hum), Precipitation Amount (Precip Amount), Wind Speed (Wind Spd), Station Pressure (Stn Press).
-   Hidden size (hidden_size)
    -   The number of hidden units in each GRU which determines the capacity of the model of capturing data patterns.
-   Output size (output_size)
    -   The number of predictions. In the case of predicting the next 24 hours of data, the output size is 24.
-   Number of layers (num_layers)
    -   The number of GRU layers in the model. The more the layer, the more complex patterns the model can learn.

#### Examples

# Data

### Data Source

The data (from 2015 to 2022) comes from [Weather Canada](https://climate.weather.gc.ca/climate_data/hourly_data_e.html?hlyRange=2009-12-10%7C2023-04-02&dlyRange=2010-02-02%7C2023-04-02&mlyRange=%7C&StationID=48549&Prov=ON&urlExtension=_e.html&searchType=stnName&optLimit=yearRange&StartYear=2015&EndYear=2022&selRowPerPage=25&Line=0&searchMethod=contains&Month=4&Day=2&txtStationName=Toronto+City+Centre&timeframe=1&Year=2023) provided by Toronto City Centre Weather Observatory.

### Data Summary and General View

Summary statistics at the scale of years usually is too extensive and vague and may not help people have a understanding of what the data really means. For example, people probably does not have picture of what a 70% relative humidity and 110 kPa means.

|      Variables      | Sample Mean | Standard Division |
| :-----------------: | :---------: | :---------------: |
|      Temp (°C)      |  9.105228   |     10.167671     |
| Dew Point Temp (°C) |  4.229453   |     10.443530     |
|     Rel Hum (%)     |  72.933690  |     15.008117     |
| Precip. Amount (mm) |  0.088654   |     0.768895      |
|   Wind Spd (km/h)   |  17.225652  |     10.397769     |
|   Stn Press (kPa)   | 100.730966  |     0.790242      |

Therefore, instead of directly output the statistics such as mean, max, min, standard deviations and so on, a visual diagram can present the trend and patterns better.

In the diagram below, it presents all the hourly weather data from 2015 to 2022. Upon comparing the temperature and wind speed diagrams, it is obvious that as the maximum wind speed increases, the temperature decreases as expected. Also, by observing the station pressure diagram, the pressure starts to fluctuate when temperatures drop and tend to be more stable in a range when the temperature rises. Additionally, the minimum relative humidity demonstrates periodic decreases in correlation with temperature, albeit with approximately one season's time difference.

![plot](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/2015-2022%20GTA%20Weather%20Data%20Plot.png?raw=true)

Example of hourly weather data on Jan 01, 2015:

![plot](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/01-01-2015%20GTA%20Weather%20Data%20Plot.png?raw=true)

### Data Processing and Transformation

Since some of the data is missing, the missing value will be replaced with the average of the two hours preceding and following the missing data point. If more than four consecutive hours of data are missing, the entire row will be dropped from the dataset. This is because if those data which filled with the average are fed to the model, then the model might lose some of the variability and will be more likely output a sequence of same prediction and is does not generalize well to the real situation.

### Data Split

The dataset is split by years in order to contain all the variability and possibility of the weather condition for four seasons.

- Training Set: Year 2015 to 2019. 

- Validation Set: Year 2020 to 2021.

- Test Set: Year 2022.

  

# Training

### Training Curve

#### Overfit Model:

> For testing the capability and correctness of the model with a small dataset.

##### Batch Learning Curve:

> The loss of batch prediction with respect to its target.

![Overfit Batch Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Overfit%20Batch%20Learning%20Curve.png?raw=true)

##### Total Learning Curve:

> The loss of the whole validation set prediction with respect to its target since the first iteration.

![Overfit Total Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Overfit%20Total%20Learning%20Curve.png?raw=true)

#### Final Model:

###### Note:

###### Since the model's loss is relatively high at the beginning and the graph of total loss will squeeze altogether. Therefore, the training curves are separated into epochs for better presentation.  #0 Epoch is not represented since the y-axis varies from loss 1.4 to 0.2 and the graph is squeezed altogether.

##### #1 Epoch

Batch Learning Curve:

> The loss of batch prediction with respect to its target.

![Epoch1 Batch Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Epoch1%20Batch%20Learning%20Curve.png?raw=true)

Epoch Learning Curve:

> The loss of the whole validation set prediction with respect to its target at the current epoch. It is the last subgraph (i.e. the tail) of the Total Learning Curve to visualize the gap between validation loss and training loss better.

![Epoch1 Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Epoch1%20Learning%20Curve.png?raw=true)

Total Learning Curve:

> The loss of the whole validation set prediction with respect to its target since the first iteration.

![Epoch1 Total Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Epoch1%20Total%20Learning%20Curve.png?raw=true)

##### #16 Epoch:

Batch Learning Curve:

> The loss of batch prediction with respect to its target.

![Epoch16 Batch Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Epoch16%20Batch%20Learning%20Curve.png?raw=true)

Epoch Learning Curve:

> The loss of the whole validation set prediction with respect to its target at the current epoch. It is the last subgraph (i.e. the tail) of the Total Learning Curve to visualize the gap between validation loss and training loss better.

![Epoch16 Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Epoch16%20Learning%20Curve.png?raw=true)

Total Learning Curve:

> The loss of the whole validation set prediction with respect to its target since the first iteration.

![Epoch16 Total Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Epoch16%20Total%20Learning%20Curve.png?raw=true)

### Regularization

#### Weight Decay:

Please refer to Hyperparameters Tuning.

#### Early Stopping:

The model is stopped at epoch #16 since the learning curve at epochs 17 and 18 started to oscillate which indicates the model converged and might start to overfit in the future.

##### #17 Epoch Learning Curve:

![Epoch17 Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Epoch17%20Learning%20Curve.png?raw=true)

##### #18 Epoch Learning Curve:

![Epoch18 Learning Curve.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Epoch18%20Learning%20Curve.png?raw=true)

### Hyperparameters Tuning

Graph of Partial Predictions of Final Model (used for comparisons to the graphs in this section).

![Partial Predictions ckpt 1428.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Partial%20Predictions%20ckpt%201428.png?raw=true)

#### Batch Size: 512

- The batch size is set to 512 to improve GPU utilization and more concurrency. Although bigger batch size may lead to limited exploration and less frequent updating weights, 512 is around 1.12% of the training set, which is considered a "small" batch size.

#### Learning Rate: 0.001

Although the loss on the graph of learning curve seems indicating the learning rate is high. The actual performance of a lower learning rate makes the model's performance worse although it can generate more .

![img](https://i.stack.imgur.com/iMASu.jpg)

While keeping all other hyperparameters unchanged, the graph below used 0.0001 as the learning rate. Clearly, although the model predicted more details (i.e. the peaks and valleys), the general trend does not fit with the target.

![Partial Predictions LR 0.0001.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/main/Images/Partial%20Predictions%20LR%200.0001.png?raw=true)

#### Momentum: 0.9

A momentum of 0.9 with a 0.001 learning rate is generally good with Adam optimizer. It is the empirical results from previous research and experimentation. Indeed, among lots of learning rate and momentum combinations tested so far in the project, a learning rate of 0.001 with 0.9 momentum is the best combination, which produces the best performance and generalizes best to the test set.

#### Weight Decay: 0.0001

The graph below shows a partial prediction with no weight decay.

![Partial Predictions.png](https://github.com/jingwenshi-dev/CSC413-Deep-Learning/blob/GRU-v1.2/Images/Partial%20Predictions.png?raw=true)

#### Number of Epochs: 16

The final model and weight is selected at the end of epoch number 16 in order to prevent overfit. Please refer to [regularization](#Regularization).

#### Number of Hidden Layers:



# Result

### Quantitative Measures

### Quantitative and Qualitative Results

### Justification of Results

# Ethical Consideration

While accurate weather predictions can benefit society, there are potential ethical implications to consider. The model's predictions may disproportionately impact specific groups, such as farmers, who rely heavily on accurate weather forecasts for their livelihoods. If the model's predictions are less accurate for specific regions or time periods, it could lead to negative consequences for these communities. Moreover, the potential misuse of the model by bad actors may lead to the dissemination of false (e.g. extreme weather conditions), causing panic or confusion. Ensuring the model's robustness, accuracy, and fairness across various regions and population groups is crucial to mitigate these ethical concerns.

# Authors

Jingwen (Steven) Shi: GRU Building & Deciding, GRU Training, GRU Hyperparameters Tuning, Result Displaying, Graph Generation, Report Writing

Hongsheng Zhong: Data Processing & Normalization, Input and Target Separation, Loss Function, Debug

Xingjian Zhang: Transformer Building & Deciding, Transformer Training, Transformer Hyperparameters Tuning

Xuankui Zhu: Scaler Research
