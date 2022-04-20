
from importlib.resources import path
import pandas as pd 
import numpy as np 
import datetime as datetime
import matplotlib.pyplot as plt 
#from fbprophet import Prophet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras import Sequential, layers, optimizers, callbacks
from tensorflow import keras
import math
from sklearn.metrics import mean_squared_error
import pickle 
import os
import itertools as it
import names
import streamlit as st
import plotly.express as px

def create_dataset(dataset, time_step=3):
    """This function creates specific numpy arrays with features and labels. 

    Args:
        dataset (numpy array): A numpy array in the shape (X, 1)
        time_step (int, optional): Number of previous days the model uses to predict the next day. Defaults to 3.

    Returns:
        Numpy arrays: Returns X in the shape (X, 3) and Y in the shape (X, 1)
    """
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def preprocess(df, country1, time_step):
    """Creates a numpy array, then split, transform and prepare the data for the timeseries model. 

    Args:
        df (Pandas Dataframe): Pandas dataframe which contains all countries data looking at confirmed cases. 
        country1 (string): Country to prepare for training.
        time_step (int): Number of previous days the model uses to predict the next day. Defaults to 3.

    Returns:
        Numpy arrays: 4 arrays ready to train and validate the model. 
    """
    df_1_cases = df[["Country/Region", country1]].copy()
    df_1_cases = df_1_cases.drop(labels=0, axis=0)
    df_1_cases.columns=["Date", "Confirmed Cases"]
    df_1_cases["Date"] = pd.to_datetime(df_1_cases["Date"], format="%m/%d/%y")
    df_1 = df_1_cases["Confirmed Cases"].to_numpy()

    df_1 = df_1.reshape(-1,1)

    training_size = int(len(df_1)*0.7) 
    
    train_data, test_data = df_1[0:training_size,:], df_1[training_size:len(df_1), :1]

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    return X_train, y_train, X_test, y_test

def model(optimizer_, learning, X_train, y_train, time_step, nr_nodes, dropout_rate, nr_layers, model_type, X_val, y_val):
    """This function trains different models based on the choices of parameters. 

    Args:
        optimizer_ (_type_): Which optimizer to use.
        learning (_type_): Learning rate for the model.
        X_train (_type_): X values to train the model.
        y_train (_type_): Y values to train the model.
        time_step (_type_): Number of previous days the model uses to predict the next day.
        nr_nodes (_type_): Number of nodes the model will have.
        dropout_rate (_type_): Dropout rate the dropout layer will use. 
        nr_layers (_type_): Number of hidden layers the model will be built on. 
        model_type (_type_): What type of model architecture. LSTM, SimpleRNN or GRU.
        X_val (_type_): X values to validate the model.
        y_val (_type_): Y_values to validate the model. 

    Returns:
        _type_: The history of the training, and the model. 
    """
    if model_type == "LSTM":
        if nr_layers == 1:
            model = Sequential()
            model.add(layers.LSTM(nr_nodes, input_shape=(time_step, 1), use_bias=False))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(1))
            opt = the_optimizer(optimizer_, learning)
            model.compile(loss="mean_squared_error", optimizer=opt)


        elif nr_layers == 2:
            model = Sequential()
            model.add(layers.LSTM(nr_nodes, input_shape=(time_step, 1), use_bias=False, return_sequences=True))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.LSTM(nr_nodes))
            model.add(layers.Dense(1))
            opt = the_optimizer(optimizer_, learning)
            model.compile(loss="mean_squared_error", optimizer=opt)

        elif nr_layers == 3:
            model = Sequential()
            model.add(layers.LSTM(nr_nodes, input_shape=(time_step, 1), use_bias=False, return_sequences=True))
            model.add(layers.LSTM(nr_nodes, return_sequences=True))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.LSTM(nr_nodes))
            model.add(layers.Dense(1))
            opt = the_optimizer(optimizer_, learning)
            model.compile(loss="mean_squared_error", optimizer=opt)
    
    elif model_type == "GRU":
        if nr_layers == 1:
            model = Sequential()
            model.add(layers.GRU(nr_nodes, input_shape=(time_step, 1), use_bias=False))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(1))
            opt = the_optimizer(optimizer_, learning)
            model.compile(loss="mean_squared_error", optimizer=opt)

        elif nr_layers == 2:
            model = Sequential()
            model.add(layers.GRU(nr_nodes, input_shape=(time_step, 1), use_bias=False, return_sequences=True))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.GRU(nr_nodes))
            model.add(layers.Dense(1))
            opt = the_optimizer(optimizer_, learning)
            model.compile(loss="mean_squared_error", optimizer=opt)

        elif nr_layers == 3:
            model = Sequential()
            model.add(layers.GRU(nr_nodes, input_shape=(time_step, 1), use_bias=False, return_sequences=True))
            model.add(layers.GRU(nr_nodes, return_sequences=True))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.GRU(nr_nodes))
            model.add(layers.Dense(1))
            opt = the_optimizer(optimizer_, learning)
            model.compile(loss="mean_squared_error", optimizer=opt)
    
    elif model_type == "SimpleRNN":
        if nr_layers == 1:
            model = Sequential()
            model.add(layers.SimpleRNN(nr_nodes, input_shape=(time_step, 1), use_bias=False))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.Dense(1))
            opt = the_optimizer(optimizer_, learning)
            model.compile(loss="mean_squared_error", optimizer=opt)

        elif nr_layers == 2:
            model = Sequential()
            model.add(layers.SimpleRNN(nr_nodes, input_shape=(time_step, 1), use_bias=False, return_sequences=True))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.GRU(nr_nodes))
            model.add(layers.Dense(1))
            opt = the_optimizer(optimizer_, learning)
            model.compile(loss="mean_squared_error", optimizer=opt)

        elif nr_layers == 3:
            model = Sequential()
            model.add(layers.SimpleRNN(nr_nodes, input_shape=(time_step, 1), use_bias=False, return_sequences=True))
            model.add(layers.SimpleRNN(nr_nodes, return_sequences=True))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.SimpleRNN(nr_nodes))
            model.add(layers.Dense(1))
            opt = the_optimizer(optimizer_, learning)
            model.compile(loss="mean_squared_error", optimizer=opt)
    else: pass

    earlystopping = callbacks.EarlyStopping(monitor="val_loss", patience = 5, restore_best_weights = True, mode = "min")
    
    return model.fit(X_train, y_train, epochs=100, batch_size=512, verbose=False, callbacks=[earlystopping], validation_data=(X_val, y_val)), model

def the_optimizer(optimizer_, learning):
    """Creates the optimizer of your choice.

    Args:
        optimizer_ (_type_): Adam or SGD
        learning (_type_): Learning rate is set to 0.001.

    Returns:
        _type_: Optimizer of choice with learning rate included. 
    """
    if optimizer_ == "Adam":
        opt = optimizers.Adam(learning_rate=learning)
    elif optimizer_ =="SGD":
        opt = optimizers.SGD(learning_rate=learning)
    else: pass
    return opt

def predict(X, ourmodel, scaler):
    """This function uses the model to predict Y values from X values. 

    Args:
        X (_type_): The values our model uses to predict on.
        ourmodel (_type_): Model in use. 
        scaler (_type_): Scaler in use

    Returns:
        _type_: Predicted values. 
    """
    pred = ourmodel.predict(X)
    pred_inverse = scaler.inverse_transform(pred)
    return pred_inverse

def predict_future(y_test, time_steps, model, no_days, scaler):
    '''Function that predicts x days ahead given data
        Args:
            y_test: Data the model uses to predict further days. 
            time_steps: Number of previous days the model uses to predict the next day.
            model: Model in use
            no_days: Number of days ahead it is predicting
            scaler: Scaler in use
        Returns:
            The predicted days ahead depending on number of days chosen to predict. 
        '''

    look_back = len(y_test) - time_steps
    look_back = int(look_back)

    x_input = y_test[look_back:].reshape(1,-1)
    
    temporary_input = list(x_input)
    temporary_input = temporary_input[0].tolist()
    output = []
    i = 0

    while(i<no_days): 
        if (len(temporary_input) > time_steps):
            x_input=np.array(temporary_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, time_steps, 1))

            y_hat = model.predict(x_input)

            temporary_input.extend(y_hat[0].tolist())
            temporary_input = temporary_input[1:]
            output.extend(y_hat.tolist())

            i = i+1
        
        else:
            x_input = x_input.reshape((1, time_steps, 1))
            y_hat = model.predict(x_input)
            temporary_input.extend(y_hat[0].tolist())
            output.extend(y_hat.tolist())
            i = i+1
    output = scaler.inverse_transform(output)
    return output

def plots(real_data, test_pred_plot, country, df_date, df_case):
    """Function that plots the real data compared to the models predicted values. 

    Args:
        real_data (_type_): Ground truth data of chosen country. 
        test_pred_plot (_type_): Predicted data of chosen country.
        country (_type_): Chosen country.

    Returns:
        _type_: Plot
    """


    

    #fig, ax = plt.subplots()
    #ax.plot(real_data, label="Real Values")
    #ax.plot(test_pred_plot, label="Predictions")
    plot_1 = plt.figure(figsize=(10,5))
    plt.plot(df_date, real_data, label="Ground truth")
    plt.plot(df_date, test_pred_plot, label="Predictions")
    plt.legend()
    plt.grid()
    plt.close()

    return plot_1

def test_new_data(df, country, scaler, our_model, time_step):
    """ Function that test the model on any given country. 

    Args:
        df (_type_): Pandas dataframe that contains the data
        country (_type_): Country of choice
        scaler (_type_): Scaler of choice
        our_model (_type_): Model of choice.
        time_step (_type_): Number of previous days the model uses to predict the next day.

    Returns:
        _type_: Predicted data of the chosen country, RMSE score of the test, Groud truth data of the chosen country. 
    """

    df_country_cases = df[["Country/Region", country]].copy()
    df_country_cases = df_country_cases.drop(labels=0, axis=0)
    df_country_cases.columns=["Date", "Confirmed Cases"]
    df_country_cases["Date"] = pd.to_datetime(df_country_cases["Date"], format="%m/%d/%y")
    df1 = df_country_cases["Confirmed Cases"].to_numpy()

    df_date = df_country_cases["Date"]
    df_date = pd.DataFrame(df_date)
    df_case = df_country_cases["Confirmed Cases"]
    df_case = pd.DataFrame(df_case)

    df1 = df1.reshape(-1,1)
    
    X1, Y1 = create_dataset(df1, time_step)
    
    X_scaled = scaler.transform(X1.reshape(-1,1))
    
    X = X_scaled.reshape(X1.shape[0], X1.shape[1], 1)

    test_pred, rmse_test = test_any_data(X, our_model, scaler, Y1)
    test_pred = test_pred.reshape(-1)
    return test_pred, rmse_test, Y1, df_date, df_case

def save_model_etc(directory_name, our_model, scaler, dictionary_, rmse_test, fig, chosen_directory, pred_plot):
    """Function that saves the model, parameters used for each model, loss and test plots, scaler, RMSE, 

    Args:
        directory_name (_type_): Randomly generated directory name where all the things saves.
        our_model (_type_): Model to save
        scaler (_type_): Scaler used
        dictionary_ (_type_): Some params for each model.
        rmse_test (_type_): RMSE for the model tested on Norway. 
        fig (_type_): Validation and train loss plot.
        chosen_directory (_type_): Parent folder where each models subfolder gets stored. 
        pred_plot (_type_): Plot of predicted and ground truth data for Norway.
    """

    parent = r"C:\Users\tobia\OneDrive\Skrivbord\ML_INTRO\ML"
    chosen_parent = os.path.join(parent, chosen_directory)
    path = os.path.join(chosen_parent, directory_name)

    try: 
        os.makedirs(path)
    except FileExistsError:
        print("BIG ERROR")

    our_model.save(path + "\model")


    with open(os.path.join(path, "dictionary.pickle"), "wb") as handle:
        pickle.dump(dictionary_, handle)
    with open(os.path.join(path, "scaler.pickle"), "wb") as handle:
        pickle.dump(scaler, handle)
    with open(os.path.join(path, "rmse_test.pickle"), "wb") as handle:
        pickle.dump(rmse_test, handle)

    pic = "loss.png"
    pic2 = "predictions.png"
    pred_plot.savefig(os.path.join(path, pic2))
    fig.savefig(os.path.join(path, pic))

def load_model_etc(directory_choice, one_or_two):
    """ This function loads all the necessities to use a specific model.

    Args:
        directory_choice (_type_): Name of the subfolder where the model is saved. 
        one_or_two (_type_): Parentfolder where all the models from a specific training is saved. 

    Returns:
        The loaded model, scaler, the params, RMSE for the test on Norway. 
    
    """

    path2 = os.path.join("Streamlit modeller", directory_choice)

    with open(os.path.join(one_or_two, directory_choice, "dictionary.pickle"), "rb") as handle:
        dictionary_ = pickle.load(handle)

    with open(os.path.join(one_or_two, directory_choice, "scaler.pickle"), "rb") as handle:
        scaler = pickle.load(handle)
    
    with open(os.path.join(one_or_two, directory_choice, "rmse_test.pickle"), "rb") as handle:
        rmse_test = pickle.load(handle)

    our_model = keras.models.load_model(path2 + "/model")
    
    return our_model, scaler, dictionary_, rmse_test

def test_any_data(X, our_model, scaler, Y1):
    """This function is used to call other funcitons to test a model on any given country. 

    Args:
        X (_type_): X values thtat the model uses to predict new values. 
        our_model (_type_): Model of choice
        scaler (_type_): Scaler attached to the model. 
        Y1 (_type_): Ground truth values. 

    Returns:
        The predicted values, and RMSE for test on chosen country.
    """
    new_country_pred = predict(X, our_model, scaler)
    rmse = math.sqrt(mean_squared_error(Y1, new_country_pred))
    return new_country_pred, rmse

def scale_merged(scaler, X_train):
    """ This function is used to scale the data before it is passed to the model. 

    Args:
        scaler (_type_): Scaler of choice.
        X_train (_type_): Data to scale.

    Returns:
        Scaled data and scaler. 
    """
    if scaler == "StandardScaler":
        scaler_ = StandardScaler()
        scaler_.fit((X_train).reshape(-1,1))
        X_train_scaled = scaler_.transform((X_train).reshape(-1,1))

    elif scaler =="MinMaxScaler":
        scaler_ = MinMaxScaler()
        scaler_.fit((X_train).reshape(-1,1))
        X_train_scaled = scaler_.transform((X_train).reshape(-1,1))

    else: pass

    X_train_scaled = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], 1)

    return X_train_scaled, scaler_

def plot_history(history):
    """This function plots the validation and train loss for the model.

    Args:
        history (_type_): Information gathered during the training of the model. 

    Returns:
        _type_: The plot of the loss.
    """
    fig = plt.figure(figsize = (10,6))
    ax = fig.add_subplot()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.close()
    return fig

def train(df, chosen_directory):
    """This is a funciton that creates our grid and then call other functions to train x number of models depending on the grid. Then also saves the model and all the necessities.

    Args:
        df (_type_): Pandas dataframe that contains all data looking at confirmed cases. 
        chosen_directory (_type_): Parentfolder where you want to save all the models from the training. 
    """
    #grid = {"Dropout": [0.1, 0.2, 0.3, 0.4], "Nodes": [10, 20, 30], "Layers": [1, 2, 3], "Scaler": ["MinMaxScaler", "StandardScaler"], "Timesteps": [2, 3, 4], "Optimizer": ["Adam", "SGD"], "Model": ["LSTM", "GRU", "SimpleRNN"]}
    grid = {"Dropout": [0.1, 0.2, 0.3, 0.4], "Nodes": [20, 30], "Layers": [1], "Scaler": ["MinMaxScaler"], "Timesteps": [12,13,14], "Optimizer": ["Adam"], "Model": ["GRU"]}
    allNames = sorted(grid)
    all_combinations = it.product(*(grid[Name] for Name in allNames))
    the_list = list(all_combinations)
    countries_train = ["Italy", "Portugal", "Greece", "Ukraine", "Spain"]

    for i in range(len(the_list)):
        save_directory = names.get_full_name()
        
        X_train_list = []
        X_val_list = []
        y_train_list = []
        y_val_list = []
        
        for country in countries_train:

            X_train_no_scale, y_train_no_scale, X_val_no_scale, y_val_no_scale = preprocess(df, country, the_list[i][6])
            X_train_list.append(X_train_no_scale)
            X_val_list.append(X_val_no_scale)
            y_train_list.append(y_train_no_scale)
            y_val_list.append(y_val_no_scale)
        
        X_train = np.concatenate(X_train_list)
        X_val = np.concatenate(X_val_list)
        y_train = np.concatenate(y_train_list)
        y_val = np.concatenate(y_val_list)


        X_train_scaled, scaler = scale_merged(the_list[i][5], X_train)
        y_train_scaled = scaler.transform((y_train).reshape(-1,1))
        X_val_scaled = scaler.transform((X_val).reshape(-1,1))
        y_val_scaled = scaler.transform((y_val).reshape(-1,1))

        X_val_scaled = X_val_scaled.reshape(X_val.shape[0], X_val.shape[1], 1)
        y_train_scaled = y_train_scaled.reshape(y_train.shape)
        y_val_scaled = y_val_scaled.reshape(y_val.shape)

        history, our_model = model(the_list[i][4], 0.001, X_train_scaled, y_train_scaled, the_list[i][6], the_list[i][3], the_list[i][0], the_list[i][1], the_list[i][2], X_val_scaled, y_val_scaled)

        fig = plot_history(history)

        test_prediction, rmse_test, Y_real, df_date = test_new_data(df, "Norway", scaler, our_model, the_list[i][6])
        
        pred_plot = plots(Y_real, test_prediction, "Norway")

        save_model_etc(save_directory, our_model, scaler, the_list[i], rmse_test, fig, chosen_directory, pred_plot)

def streamlit_test(plot, rmse):
    st.plotly_chart(plot)
    st.write("The RMSE value for this model tested on Norway is: ", rmse)

def streamlit_forecast(plot):
    st.write("Hejsans")
    fig = plt.figure(figsize=(10,5))
    plt.plot(plot, label="Predicted number of cases")
    plt.ylabel("Number of cases in norway")
    plt.xlabel("Days ahead, starting at 9th of february 2022")
    plt.grid()
    plt.legend()
    st.plotly_chart(fig)

