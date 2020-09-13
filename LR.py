import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

np.random.seed(42)


class Scaler():
    def __init__(self):
        self.min_train = 0
        self.max_train = 0
        self.std_train = 0
        self.mean_train = 0
    def __call__(self,features, is_train=False):
        if (is_train):

            self.std_train = features.std(axis=0)
            self.mean_train = features.mean(axis=0)
            standardize_df = (features - self.mean_train) / self.std_train
            self.min_train = standardize_df.min(axis=0)
            self.max_train = standardize_df.max(axis=0)
            normalized_df = (standardize_df - self.min_train) / (
                    self.max_train - self.min_train)

        else:
            standardize_df = (features - self.mean_train) / self.std_train
            normalized_df = (standardize_df - self.min_train) / (
                    self.max_train - self.min_train)
        return normalized_df


def get_features(csv_path,is_train=False,scaler=None):
    df_feature = pd.read_csv(csv_path, sep=r'\s*,\s*', engine='python')
    if (is_train):
        for column in df_feature.columns:
            corr = df_feature[column].corr(df_feature['shares'])
            if (np.abs(corr) < 1e-3):
                drop_column.append(column)
        df_feature.drop(drop_column, axis='columns', inplace=True)
    else:
        df_feature.drop(drop_column, axis='columns', inplace=True)
    if 'shares' in df_feature:
        df_feature.drop('shares',axis='columns', inplace=True)
    df_feature=scaler(df_feature,is_train)
    feature_bias = np.zeros(len(df_feature.index))
    feature_bias.fill(1)
    df_feature.insert(loc=0, column='bias', value=feature_bias)
    return df_feature.to_numpy()

def get_targets(csv_path):
    df_feature = pd.read_csv(csv_path, sep=r'\s*,\s*', engine='python')
    return df_feature['shares'].to_numpy()
     

def analytical_solution(feature_matrix, targets, C=0.0001):
    x=feature_matrix.transpose().dot( feature_matrix)
    y=np.linalg.inv(x+C*np.identity(x.shape[0]))
    w= y.dot(feature_matrix.T).dot(targets)
    return w

def get_predictions(feature_matrix, weights):
    predictions = feature_matrix.dot(weights)
    return predictions

def mse_loss(feature_matrix, weights, targets):
    prediction=get_predictions(feature_matrix, weights)
    error = targets - prediction
    return error.T.dot(error) / len(targets)

def l2_regularizer(weights):
    return weights.dot(weights)

def loss_fn(feature_matrix, weights, targets, C=0.0001):
    return mse_loss(feature_matrix,weights,targets)+C*l2_regularizer(weights)

def compute_gradients(feature_matrix, weights, targets, C=0.0001):
    l2_gradient = 2 * C * weights
    mse_gradient = 2 * (feature_matrix.T.dot(feature_matrix).dot(weights) - feature_matrix.T.dot(targets))/ len(targets)
    return mse_gradient+l2_gradient

def sample_random_batch(feature_matrix, targets, batch_size):
    batch=np.random.choice(feature_matrix.shape[0], batch_size, replace=False)
    sampled_feature_matrix=feature_matrix[batch, :]
    sampled_targets =targets[batch]
    return (sampled_feature_matrix, sampled_targets)
    
def initialize_weights(n):
    weights = np.zeros(n)
    #weights.fill(0)
    return weights

def update_weights(weights, gradients, lr):
    weights=weights-lr*gradients
    return weights

def early_stopping(dev_loss, weights, patience=5):
    if (early_stopping.min_dev_loss < dev_loss):
        early_stopping.count += 1
    else:
        early_stopping.count = 0
        early_stopping.min_dev_loss = dev_loss
        early_stopping.min_weights = weights
    if (early_stopping.count >= patience):
        return True
    return False
    

def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=0.1,
                        C=1e-7,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=500):
    weights = initialize_weights(train_feature_matrix.shape[1])
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    early_stopping.count = 0
    early_stopping.min_dev_loss = dev_loss
    early_stopping.min_weights = weights
    #these are arrays maintained to plot the graph
    dev_loss_array = []
    train_loss_array = []
    plotindex = []

    #early stopping parameter
    patience=10
    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        gradients = compute_gradients(features, weights, targets, C)
        weights = update_weights(weights, gradients, lr)

        if step%eval_steps == 0:
            dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
            train_loss = mse_loss(train_feature_matrix, weights, train_targets)
            if (early_stopping(dev_loss, weights, patience)):
                break
            if (dev_loss < 1000):
                plotindex.append(step)
                dev_loss_array.append(dev_loss)
                train_loss_array.append(train_loss)
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))
    #plot graph for dev/test loss against steps
    plt.title("dev, test Loss v/s steps")
    plt.xlabel("steps")
    plt.ylabel("dev/test Loss")
    plt.text(50000, 600, r'Dev_Loss', color='blue')
    plt.text(50000, 800, r'Train_Loss', color='green')
    plt.plot(plotindex, dev_loss_array, color="blue")
    plt.plot(plotindex, train_loss_array, color="green")
    plt.show()
    weights = early_stopping.min_weights
    return weights

def do_evaluation(feature_matrix, targets, weights):
    #predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

if __name__ == '__main__':
    scaler = Scaler()
    drop_column = []
    train_features, train_targets = get_features('data/train.csv',True,scaler), get_targets('data/train.csv')
    dev_features, dev_targets = get_features('data/dev.csv',False,scaler), get_targets('data/dev.csv')

    a_solution = analytical_solution(train_features, train_targets, C=0.0001)
    train_predictions=get_predictions(train_features, a_solution)
    # plot graph against predicted shares and target shares
    plt.title("prediction v/s targets")
    plt.xlabel("predictions")
    plt.ylabel("targets")
    plt.plot(train_predictions,train_targets,'o' ,color='green')
    plt.show()
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))

    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features, 
                        train_targets, 
                        dev_features,
                        dev_targets,
                        lr=0.1,
                        C=1e-7,
                        batch_size=32,
                        max_steps=2000000,
                        eval_steps=500)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    print('Increase patience parameter in gradient descent funtion to improve results\n')
    test_features = get_features('data/test.csv', False, scaler)
    test_predictions=get_predictions(test_features, a_solution)
    prediction_df = pd.DataFrame(data=test_predictions, columns=["shares"])
    prediction_df.index.name = 'instance_id'
    prediction_df.to_csv("test_prediction_shares.csv")
    

