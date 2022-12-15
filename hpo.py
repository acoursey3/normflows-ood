import pandas as pd 
import numpy as np
import normflows as nf
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from itertools import product

def load_train_data():
    training_data = pd.read_csv('fault_free_training.csv')
    training_data.drop(['faultNumber', 'sample', 'Unnamed: 0'], axis=1, inplace=True)
    training_data.rename(columns={'simulationRun': 'id'}, inplace=True)

    X_train = training_data.drop(['id'], axis=1)
    scaler = MinMaxScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)

    del training_data
    X_train = np.array(X_train)
    return X_train, scaler

# Adds time lag to data to give time context to each data point
def lag_data(data, n_lag):
    return series_to_supervised(data, n_in=n_lag, dropnan=False).fillna(0).to_numpy()

# Source: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Takes lagged predictions and returns the "unlagged" predictions (unused)
def unlag_predictions(preds, n_lag):
    return preds.detach().numpy()[:,-52:]

def create_flow_model(config, data_shape):
    base = nf.distributions.base.DiagGaussian(data_shape)

    # Define list of flows
    num_layers = config['num_flows']
    flows = []
    for i in range(num_layers):
        param_map = nf.nets.MLP([math.floor(data_shape/2), config['n_neurons'], config['n_neurons'], 
                                 data_shape], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        flows.append(nf.flows.Permute(2, mode='swap'))

    model = nf.NormalizingFlow(base, flows)
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)
    
    return model

# Predicts which points are anomalies given a dataset and a model
def predict_anomalies(model, data, scaler, prob_scaler, threshold, lagged, training=False):
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    
    if not training:
        category = data['faultNumber']
        data['faultNumber'] = data['sample'] > 20 # Faults were introduced one hour in (20 samples)
        faults = data['faultNumber']
    else:
        category = 0
        faults = [0] * len(data['faultNumber'])
    data.drop(['faultNumber', 'sample', 'Unnamed: 0'], axis=1, inplace=True)
    data.rename(columns={'simulationRun': 'id'}, inplace=True)
    
    data = data.drop(['id'], axis=1)
    data = np.array(data)
    data = scaler.transform(data)
    
    if lagged:
        data = lag_data(data, 3)
    
    model.eval()
    test_probs = np.nan_to_num(model.log_prob(torch.tensor(data).float().to(device)).detach().numpy(), nan=-9999, neginf=-9999)
    test_probs = prob_scaler.transform(test_probs.reshape(-1,1))

    anom_preds = test_probs < threshold
    
    return faults, anom_preds.tolist(), test_probs.tolist(), category

# Runs accuracy of a trial on one set of hyperparameters
def hyperparameter_trial(X_train, config, scaler):
    
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    
    if config['lag']:
        X_train = lag_data(X_train, 3)
    
    model = create_flow_model(config, X_train.shape[1])
    
    max_iter = config['epochs']
    num_samples = config['batch']

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    # Trains model
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()

        sample_indices = np.random.choice(range(X_train.shape[0]), num_samples, replace=False)
        x = torch.tensor(X_train[sample_indices,:]).float().to(device)

        loss = model.forward_kld(x)

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

    # Predictions using model
    model.eval()
    # This is using up a lot of memory - can we split it into batches?
    log_prob = model.log_prob(torch.tensor(X_train).float().to(device))

    del X_train
    
    prob = log_prob
    del log_prob
    
    prob = prob.detach().numpy()
    prob = np.nan_to_num(prob, nan=-9999, neginf=-9999, posinf=9999)
    prob_scaler = MinMaxScaler()
    scaled_train_probs = prob_scaler.fit_transform(prob.reshape(-1, 1))
    del prob
    threshold = np.median(scaled_train_probs) - 3 * np.std(scaled_train_probs)
    
    y = []
    yhat = []
    anomaly_probs = []
    categories = []

    with pd.read_csv('faulty_training.csv', chunksize=50000) as reader:
        for chunk in reader:
            curr_y, curr_yhat, curr_probs, curr_categories = predict_anomalies(model, chunk, scaler, prob_scaler, threshold, config['lag'])
            y.extend(curr_y)
            yhat.extend(curr_yhat)
            anomaly_probs.extend(curr_probs)
            categories.extend(curr_categories)

    yhat = np.array(yhat).squeeze()
    anomaly_probs = np.array(anomaly_probs).squeeze()
    
    y_train = []
    yhat_train = []
    anomaly_probs_train = []

    with pd.read_csv('fault_free_testing.csv', chunksize=50000) as reader:
        for chunk in reader:
            curr_y, curr_yhat, curr_probs, _ = predict_anomalies(model, chunk, scaler, prob_scaler, threshold, config['lag'], training=True)
            y_train.extend(curr_y)
            yhat_train.extend(curr_yhat)
            anomaly_probs_train.extend(curr_probs)

    yhat_train = np.array(yhat_train).squeeze()
    anomaly_probs_train = np.array(anomaly_probs_train).squeeze()
    # Is this necessary?
    anomaly_probs_train = np.clip(anomaly_probs_train, 0, 1)
    
    # Combine the predictions on the two (faulty and nominal) test datasets
    y.extend(y_train)
    yhat = yhat.tolist()
    yhat.extend(yhat_train.tolist())
    anomaly_probs = anomaly_probs.tolist()
    anomaly_probs.extend(anomaly_probs_train.tolist())
    
    acc = accuracy_score(y,yhat)
    
    return acc

if __name__=="__main__":
    X_train, scaler = load_train_data()

    # Define the grid to search over
    config = {
        'epochs': [500, 1000, 2500],
        'batch': [2**8, 2**9, 2**10],
        'lag': [True, False],
        'num_flows': [16, 32],
        'n_neurons': [64],
        'lr': [4e-4, 4e-5, 4e-6]
    }

    hyperparameter_df = pd.DataFrame(columns=['accuracy', 'epochs', 'batch', 'lag', 'num_flows', 'n_neurons', 'lr'])
    # https://stackoverflow.com/questions/64645075/how-to-iterate-through-all-dictionary-combinations
    keys, values = zip(*config.items())
    configurations = [dict(zip(keys, p)) for p in product(*values)]

    # Runs grid search
    for trial_hyp in configurations:
        acc = hyperparameter_trial(X_train, trial_hyp, scaler)
        row = trial_hyp
        row['accuracy'] = acc
        # https://stackoverflow.com/questions/51774826/append-dictionary-to-data-frame
        df_dictionary = pd.DataFrame([row])
        hyperparameter_df = pd.concat([hyperparameter_df, df_dictionary], ignore_index=True)
        print("Completed trial", df_dictionary)

    print(hyperparameter_df)

    hyperparameter_df.to_csv("./hyps.csv")