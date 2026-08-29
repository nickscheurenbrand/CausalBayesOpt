import argparse
import math
import os
from collections import namedtuple
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import torch
import wandb
from torch import nn

Data = namedtuple("Data", ["samples", "nodes"])

p = 0
n_common = 64
common_permutation = None


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, n_common),
            nn.Tanh(),
            nn.Linear(n_common, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.layers(x)


def get_net(in_features):
    n_common = 200
    if model_type == "model_linear":
        net = nn.Sequential(nn.Linear(in_features, 1))
    else:
        net = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(in_features, n_common),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, n_common),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(n_common, 1),
        )
    net.apply(
        lambda m: (m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    )
    return net


def SEM_function_linear(parents, a=1, b=0):
    return a * torch.sum(parents, dim=1, keepdim=True) + b


def SEM_function_log_sum_exp(parents, a=1, b=0):
    return a * torch.log(torch.sum(torch.exp(parents), dim=1, keepdim=True)) + b


def SEM_function_sqrt_sum(parents, a=1, b=0):
    return a * torch.sqrt(torch.sum(torch.abs(parents), dim=1, keepdim=True)) + b


def generate_multivariate_time_series(
    n_data_points,
    n_variables,
    lag,
    fixed_functions,
    connectivity_matrix,
    noise_stddev=1,
):
    """
    Generates a multivariate time series dataset based on the given parameters.

    Parameters:
    n_data_points (int): Number of data points (time steps) in the time series.
    n_variables (int): Number of variables in the multivariate time series.
    lag (int): Number of previous time steps that every variable's parents can belong to.
    fixed_functions (list): List of MLPs, one for each variable.
    connectivity_matrix (np.array): Binary connectivity matrix (lag x n_variables x n_variables) representing the causal structure.
    noise_stddev (float): Standard deviation of the Gaussian noise. Default is 1.

    Returns:
    torch.Tensor: A PyTorch tensor (n_data_points x n_variables) containing the generated multivariate time series data.
    """

    SCALE = 10
    # Initialize the multivariate time series data
    data = torch.zeros((n_data_points, n_variables))
    data[:lag, :] = SCALE * (2 * torch.rand(lag, n_variables) - 1)

    # Generate the data for each time step
    for t in range(lag, n_data_points):
        for i in range(n_variables):
            parent_input = torch.tensor(
                [
                    data[t - l - 1, j]
                    for l in range(lag)
                    for j in range(n_variables)
                    if connectivity_matrix[l, j, i] == 1
                ]
            )
            data[t, i] = SCALE * fixed_functions[i](
                parent_input.unsqueeze(0)
            ).item() + np.random.normal(scale=noise_stddev)

    return data


def generate_target_time_series(time_series_data, adjacency_matrix, transform, nsr=0):
    """
    Generates a target time series of a single variable based on the input time series and a binary adjacency matrix.

    Parameters:
    time_series_data (torch.Tensor): Input time series data (n_data_points x n_variables).
    adjacency_matrix (np.array): Binary adjacency matrix (lag x n_variables) representing the parents of the target variable.
    transform:  Fct that takes as input a 2d torch array and produces as output a 2d torch array with last dim=1 and same outer dim
    nsr (float): Noise to Signal ration. Default is 0.

    Returns:
    torch.Tensor: A PyTorch tensor (n_data_points x 1) containing the generated target time series data.
    """

    n_data_points, n_variables = time_series_data.shape
    lag = adjacency_matrix.shape[0]
    n_parents = adjacency_matrix.sum()

    if n_parents == 0:
        n_parents = 1
    parent_data = torch.zeros((n_data_points, n_parents))
    # Generate the parent data for each time step
    p = 0
    for l in range(lag):
        for j in range(n_variables):
            if adjacency_matrix[l, j] == 1:
                parent_data[lag:, p] = time_series_data[
                    lag - 1 - l : n_data_points - 1 - l, j
                ]
                p += 1
    with torch.no_grad():
        target_data = transform(parent_data)
    target_data += math.sqrt(nsr) * target_data.std() * torch.randn_like(target_data)

    return target_data


# Define the target variable's binary adjacency matrix
def construct_connectivity_matrix_target(lag, n_variables, pc):
    return (
        (np.random.rand(lag * n_variables) < pc).astype(int).reshape((lag, n_variables))
    )


# Define the connectivity matrix
def construct_connectivity_matrix(lag, n_variables, pc):
    matrix = (
        (np.random.rand(lag * n_variables * n_variables) < pc)
        .astype(int)
        .reshape((lag, n_variables, n_variables))
    )
    for i in range(n_variables):
        if matrix[:, :, i].sum() == 0:
            l, s = np.random.randint(0, lag), np.random.randint(0, n_variables)
            matrix[l, s, i] = 1
    return matrix


def construct_total_connectivity_matrix(
    connectivity_matrix, connectivity_matrix_target
):
    lag, n_variables = connectivity_matrix_target.shape
    total_connectivity_matrix = np.zeros(
        (lag, n_variables + 1, n_variables + 1), dtype=int
    )
    total_connectivity_matrix[:, :n_variables, :n_variables] = connectivity_matrix
    total_connectivity_matrix[:, :-1, -1] = connectivity_matrix_target
    return np.concatenate(
        [
            np.zeros((1, n_variables + 1, n_variables + 1), dtype=int),
            total_connectivity_matrix,
        ]
    )


def create_dataset(
    n_series, n_data_points, n_variables, lag, pc, transform_type, a, nsr
):
    connectivity_matrix = construct_connectivity_matrix(lag, n_variables, pc)
    connectivity_matrix_target = construct_connectivity_matrix_target(
        lag, n_variables, pc
    )

    # Define the MLPs
    fixed_functions = [
        MLP(connectivity_matrix[:, :, i].sum()) for i in range(n_variables)
    ]
    for mlp in fixed_functions:
        mlp.apply(
            lambda m: (
                nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None
            )
        )

    if transform_type == "logsumexp":
        transform = lambda pars: SEM_function_log_sum_exp(pars, a=a, b=0)
    elif transform_type == "linear":
        transform = lambda pars: SEM_function_linear(pars, a=a, b=0)
    elif transform_type == "sqrt":
        transform = lambda pars: SEM_function_sqrt_sum(pars, a=a, b=0)
    else:  # transform_type=='MLP'
        n_parents = connectivity_matrix_target.sum()
        net = MLP(n_parents if n_parents > 0 else 1)
        net.apply(
            lambda m: (
                nn.init.xavier_uniform_(m.weight) if isinstance(m, nn.Linear) else None
            )
        )
        transform = lambda pars: a * net(pars)

    Xs, ys = [], []
    for i in range(n_series):
        time_series_data = generate_multivariate_time_series(
            n_data_points, n_variables, lag, fixed_functions, connectivity_matrix
        )
        time_series_target = generate_target_time_series(
            time_series_data, connectivity_matrix_target, transform, nsr
        )
        Xs.append(time_series_data)
        ys.append(time_series_target)
    total_connectivity_matrix = construct_total_connectivity_matrix(
        connectivity_matrix, connectivity_matrix_target
    )
    X, y = torch.stack(Xs), torch.stack(ys)
    return X, y, total_connectivity_matrix


loss = nn.MSELoss()


def log_rmse(net, features, targets):
    net.eval()
    with torch.no_grad():
        rmse = torch.sqrt(loss(net(features), targets))
    return rmse.item()


def standard_normalization(X):
    means = X.mean(axis=0, keepdims=True)
    stds = X.std(axis=0, keepdims=True)
    normalized = (X - means) / stds
    return means, stds, normalized


def my_load_array(X, y, batch_size):
    dataset = torch.utils.data.dataset.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)


def train(
    net,
    train_features,
    train_labels,
    test_features,
    test_labels,
    num_epochs,
    learning_rate,
    l1_reg,
    l2_reg,
    batch_size,
    early_stopping,
    closed_form,
):

    if closed_form:
        X, y = torch.cat([train_features, test_features]), torch.cat(
            [train_labels, test_labels]
        )
        # XTX=torch.linalg.matmul(X.T,X)
        # XTX_inv=torch.linalg.inv(XTX)
        # A=torch.linalg.matmul(XTX_inv,X.T)
        # b=torch.linalg.matmul(A,y)
        # b=torch.linalg.pinv(X)@y
        b = torch.linalg.lstsq(X, y, driver="gels").solution
        with torch.no_grad():
            (list(net.parameters())[1]).data[:] = 0
            (list(net.parameters())[0]).data[:] = b.T
            return

    train_iter = my_load_array(train_features, train_labels, batch_size)
    # The Adam optimization algorithm is used here
    optimizer = torch.optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=l2_reg
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    best_test_loss = float("inf")
    rounds_no_improve = 0  # counter for the number of rounds without improvement

    # XXX this is the important loop that is necessary for understanding
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l += l1_reg * sum(
                param.norm(p=1)
                for name, param in net.named_parameters()
                if "weight" in name
            )
            l.backward()
            optimizer.step()
        if test_labels is not None:
            test_loss = log_rmse(net, test_features, test_labels)
            # Check if validation error has improved, and stop early if it has not improved for "early_stopping" rounds
            if early_stopping and test_loss < best_test_loss:
                best_test_loss = test_loss
                rounds_no_improve = 0
            else:
                rounds_no_improve += 1
                if rounds_no_improve == early_stopping:
                    return
            scheduler.step(test_loss)
    return


def cross_fit(
    X,
    y,
    intervened_nodes,
    k,
    num_epochs,
    learning_rate,
    l1_reg,
    l2_reg,
    batch_size,
    closed_form,
):
    global common_permutation
    # Shuffle the data using a random permutation
    perm = common_permutation
    X = X[perm]
    y = y[perm]

    # trying to see if removing some samples from the training helps
    intervened_nodes = intervened_nodes[perm]

    # Calculate sizes for each fold
    fold_sizes = [len(X) // k] * k
    for i in range(len(X) % k):
        fold_sizes[i] += 1

    # Split the data into k folds
    folds_X = torch.split(X, fold_sizes)
    folds_y = torch.split(y, fold_sizes)
    folds_intervened_nodes = torch.split(intervened_nodes, fold_sizes)

    y_preds = []
    indices = []
    for i in range(k):
        # XXX dont include samples that were part of another
        # Use the ith fold for validation, and concatenate the other k-1 folds for training
        test_X, test_y, test_intervened_nodes = (
            folds_X[i],
            folds_y[i],
            folds_intervened_nodes[i],
        )
        train_X = torch.cat([f for j, f in enumerate(folds_X) if j != i])
        train_y = torch.cat([f for j, f in enumerate(folds_y) if j != i])

        # XXX picking which samples to not include in the dataset
        train_intervened_nodes = torch.cat(
            [f for j, f in enumerate(folds_intervened_nodes) if j != i]
        )
        all_false_train = torch.all(train_intervened_nodes == False, dim=1)
        all_false_test = torch.all(test_intervened_nodes == False, dim=1)
        # include_test_samples = (test_intervened_nodes[:, i] | all_false_test).reshape(
        #     -1
        # )

        # include_train_samples = (
        #     train_intervened_nodes[:, i] | all_false_train
        # ).reshape(-1)

        # redefine the folds
        # test_X, test_y = test_X[include_test_samples], test_y[include_test_samples]
        # train_X, train_y = (
        #     train_X[include_train_samples],
        #     train_y[include_train_samples],
        # )

        # Split the training data into training and validation sets
        num_train = len(train_X)
        split_idx = int(num_train * 0.8)  # 80% training, 20% validation
        perm_train = torch.randperm(num_train)
        train_X, train_y, val_X, val_y = (
            train_X[perm_train[:split_idx]],
            train_y[perm_train[:split_idx]],
            train_X[perm_train[split_idx:]],
            train_y[perm_train[split_idx:]],
        )

        # Create the neural network model
        net = get_net(X.shape[1])

        # Train the model using early stopping
        train(
            net,
            train_X,
            train_y,
            val_X,
            val_y,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            batch_size=batch_size,
            early_stopping=20,
            closed_form=closed_form,
        )

        # Make predictions on the validation fold using the trained model
        with torch.no_grad():
            net.eval()
            n_dim = test_X.shape[-1]
            means = test_X.mean(dim=0)
            test_X = test_X.unsqueeze(1).expand(-1, n_dim, -1).clone()
            for i in range(1, n_dim):
                # test_X[:,i,i]=means[i]
                test_X[:, i, i] = 0
            y_pred = net(test_X)
            y_preds.append(y_pred)

    # Concatenate the predicted y values from each fold and return them
    return torch.cat(y_preds)[torch.argsort(perm)]


# def infer_causal_parents(
#     path,
#     num_epochs,
#     learning_rate,
#     l1_reg,
#     l2_reg,
#     batch_size,
#     train_double,
#     N_data,
#     closed_form,
#     use_wandb,
# ):
#     global common_permutation

#     # split the data into the features, the target
#     # X, y, groundtruth, data_conf = get_data(path, N_data)
#     X, y, groundtruth, data_conf = get_data_new(path, N_data)
#     node_names = groundtruth.index

#     experiment_config = {
#         "train_double": train_double,
#         "closed_form": closed_form,
#         "num_epochs": num_epochs,
#         "learning_rate": learning_rate,
#         "l1_reg": l1_reg,
#         "l2_reg": l2_reg,
#         "batch_size": batch_size,
#         "model_type": model_type,
#         "algorithm": "SITF",
#     }
#     experiment_config.update(data_conf)
#     if use_wandb:
#         run = wandb.init(
#             project="NEURIPS_DRCFS_TIMESERIES",
#             name="SR"
#             + experiment_config["model_type"]
#             + "_"
#             + experiment_config["dataset_type"]
#             + "_"
#             + experiment_config["dataset_transform_type"]
#             + "_feat_"
#             + str(experiment_config["N_feat"])
#             + "_n_obs_"
#             + str(X.shape[0])
#             + "_a_"
#             + str(experiment_config["data_coef_a"])
#             + "_nsr_"
#             + str(experiment_config["data_nsr"]),
#             config=experiment_config,
#         )

#     # y_true=a*(X[:,list(np.nonzero(groundtruth.to_numpy())[0]+1)]).sum(dim=1).reshape(-1,1)
#     # y_true=experiment_config['data_coef_a']*(X[:,list(groundtruth.to_numpy().nonzero()[0]+1)].exp().sum(dim=1)).log().reshape(-1,1)
#     def mean_and_std(z):
#         print(
#             f"inference mean: {z.mean().item():.3f} inference std: {z.std().item():.3f}"
#         )

#     def calculate_accuracy(a, b):
#         a, b = a.to_numpy()[0], b.to_numpy()
#         z = a - b
#         res = np.abs(z).sum() / len(z)
#         return 1 - res

#     def ttest(y, a0, r0, ai, ri, method):
#         if method == "direct":
#             z0, zi = y * r0, y * ri
#         else:
#             z0, zi = y * r0 - a0 * (r0 - y), y * ri - ai * (ri - y)
#             z = zi - z0
#             print(f"ttest mean: {z.mean().item():.3f} ttest std: {z.std().item():.3f}")
#         return stats.ttest_rel(z0, zi)

#     methods = ["direct", "dr"]
#     causal_graph = {
#         method: pd.DataFrame([{node_name: 0 for node_name in node_names}])
#         for method in methods
#     }
#     pValues = {
#         method: pd.DataFrame([{node_name: 0 for node_name in node_names}])
#         for method in methods
#     }
#     confidence_level = 0.95
#     Nfolds = 5
#     common_permutation = torch.randperm(X.shape[0])
#     a0 = cross_fit(
#         X, y, Nfolds, num_epochs, learning_rate, l1_reg, l2_reg, batch_size, closed_form
#     )
#     r0 = (
#         cross_fit(
#             X,
#             y,
#             Nfolds,
#             num_epochs,
#             learning_rate,
#             l1_reg,
#             l2_reg,
#             batch_size,
#             closed_form,
#         )
#         if train_double
#         else a0
#     )
#     z1, z2 = a0[:, 0] - y, r0[:, 0] - y
#     print("y std:", y.std().item())
#     # print('y_true std:',y_true.std().item())
#     mean_and_std(z1)
#     mean_and_std(z2)
#     for i in range(1, X.shape[1]):
#         print(node_names[i - 1], " groundtruth ", groundtruth.iloc[i - 1])
#         # ai=cross_fit(torch.cat([X[:,0:i],X[:,i+1:]],dim=1),y,Nfolds,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,closed_form)
#         # ri=cross_fit(torch.cat([X[:,0:i],X[:,i+1:]],dim=1),y,Nfolds,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,closed_form) if train_double else ai
#         z1, z2 = a0[:, i] - y, r0[:, i] - y
#         mean_and_std(z1)
#         mean_and_std(z2)
#         for method in methods:
#             test_res = ttest(y, a0[:, 0], r0[:, 0], a0[:, i], r0[:, i], method)
#             pValues[method][node_names[i - 1]] = (test_res[1]).item()
#             if method == "dr":
#                 print(node_names[i - 1], method, test_res)
#     print("______Corrected p values______")
#     for method in methods:
#         pValues[method].iloc[0, :] = sm.stats.multipletests(
#             pValues[method].iloc[0, :], alpha=1 - confidence_level, method="fdr_by"
#         )[1]
#         causal_graph[method] = pValues[method].applymap(
#             lambda p: int(p < (1 - confidence_level))
#         )
#     df = (
#         pd.DataFrame(groundtruth)
#         .rename(columns={"Y": "groundtruth"})
#         .join((causal_graph["dr"]).T.rename(columns={0: "dr_causal_graph"}))
#         .join((pValues["dr"]).T.rename(columns={0: "dr_p_values"}))
#         .join((causal_graph["direct"]).T.rename(columns={0: "direct_causal_graph"}))
#         .join((pValues["direct"]).T.rename(columns={0: "direct_p_values"}))
#     )
#     print(df.to_string())
#     acc = calculate_accuracy(causal_graph["dr"], groundtruth)
#     false_positives, false_negatives = sum(
#         ((causal_graph["dr"]).T)[0][groundtruth == 0]
#     ), -sum(((causal_graph["dr"]).T)[0][groundtruth == 1] - 1)
#     gdt_positives, gdt_negatives = sum(groundtruth == 1), sum(groundtruth == 0)

#     print(
#         f'N_features: {experiment_config["N_feat"]}, N_data: {experiment_config["N_obs"]}'
#     )
#     print("accuracy: ", acc)
#     print(
#         "groundtruth positives: ", gdt_positives, "false negatives: ", false_negatives
#     )
#     print(
#         "groundtruth negatives: ", gdt_negatives, "false positives: ", false_positives
#     )

#     if use_wandb:
#         wandb.log({"groundtruth_positives": gdt_positives})
#         wandb.log({"groundtruth_negatives": gdt_negatives})
#         wandb.log({"false_negatives": false_negatives})
#         wandb.log({"false_positives": false_positives})
#         run.finish()


def infer_causal_parents_new(
    X,
    y,
    intervened_nodes,
    groundtruth,
    data_conf,
    num_epochs,
    learning_rate,
    l1_reg,
    l2_reg,
    batch_size,
    train_double,
    N_data,
    var_names,
    closed_form,
    use_wandb,
):
    global common_permutation
    # split the data into the features, the target
    # X, y, groundtruth, data_conf = get_data(path, N_data)
    # X, y, groundtruth, data_conf = get_data_new(path, N_data)

    # need to figure out what each of these columns mean
    node_names = groundtruth.index
    # node_names = var_names

    experiment_config = {
        "train_double": train_double,
        "closed_form": closed_form,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "batch_size": batch_size,
        "model_type": model_type,
        "algorithm": "SITF",
    }
    experiment_config.update(data_conf)
    if use_wandb:
        run = wandb.init(
            project="NEURIPS_DRCFS_TIMESERIES",
            name="SR"
            + experiment_config["model_type"]
            + "_"
            + experiment_config["dataset_type"]
            + "_"
            + experiment_config["dataset_transform_type"]
            + "_feat_"
            + str(experiment_config["N_feat"])
            + "_n_obs_"
            + str(X.shape[0])
            + "_a_"
            + str(experiment_config["data_coef_a"])
            + "_nsr_"
            + str(experiment_config["data_nsr"]),
            config=experiment_config,
        )

    # y_true=a*(X[:,list(np.nonzero(groundtruth.to_numpy())[0]+1)]).sum(dim=1).reshape(-1,1)
    # y_true=experiment_config['data_coef_a']*(X[:,list(groundtruth.to_numpy().nonzero()[0]+1)].exp().sum(dim=1)).log().reshape(-1,1)
    def mean_and_std(z):
        print(
            f"inference mean: {z.mean().item():.3f} inference std: {z.std().item():.3f}"
        )

    def calculate_accuracy(a, b):
        a, b = a.to_numpy()[0], b.to_numpy()
        z = a - b
        res = np.abs(z).sum() / len(z)
        return 1 - res

    def ttest(y, a0, r0, ai, ri, method):
        if method == "direct":
            z0, zi = y * r0, y * ri
        else:
            z0, zi = y * r0 - a0 * (r0 - y), y * ri - ai * (ri - y)
            z = zi - z0
            print(f"ttest mean: {z.mean().item():.3f} ttest std: {z.std().item():.3f}")
        return stats.ttest_rel(z0, zi)

    methods = ["direct", "dr"]
    causal_graph = {
        method: pd.DataFrame([{node_name: 0 for node_name in node_names}])
        for method in methods
    }
    pValues = {
        method: pd.DataFrame([{node_name: 0 for node_name in node_names}])
        for method in methods
    }
    confidence_level = 0.95
    Nfolds = 5
    common_permutation = torch.randperm(X.shape[0])
    a0 = cross_fit(
        X,
        y,
        intervened_nodes,
        Nfolds,
        num_epochs,
        learning_rate,
        l1_reg,
        l2_reg,
        batch_size,
        closed_form,
    )
    r0 = (
        cross_fit(
            X,
            y,
            intervened_nodes,
            Nfolds,
            num_epochs,
            learning_rate,
            l1_reg,
            l2_reg,
            batch_size,
            closed_form,
        )
        if train_double
        else a0
    )
    z1, z2 = a0[:, 0] - y, r0[:, 0] - y
    print("y std:", y.std().item())
    # print('y_true std:',y_true.std().item())
    mean_and_std(z1)
    mean_and_std(z2)
    for i in range(1, X.shape[1]):
        print(node_names[i - 1], " groundtruth ", groundtruth.iloc[i - 1])
        # ai=cross_fit(torch.cat([X[:,0:i],X[:,i+1:]],dim=1),y,Nfolds,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,closed_form)
        # ri=cross_fit(torch.cat([X[:,0:i],X[:,i+1:]],dim=1),y,Nfolds,num_epochs, learning_rate, l1_reg,l2_reg, batch_size,closed_form) if train_double else ai

        # XXX decipher a bit which values to look at
        all_false = torch.all(intervened_nodes == False, dim=1)
        include_sample = (intervened_nodes[:, i - 1] | all_false).reshape(-1)

        # z1, z2 = (
        #     a0[include_sample, i] - y[include_sample],
        #     r0[include_sample, i] - y[include_sample],
        # )
        # mean_and_std(z1)
        # mean_and_std(z2)
        # for method in methods:
        #     test_res = ttest(
        #         y[include_sample],
        #         a0[include_sample, 0],
        #         r0[include_sample, 0],
        #         a0[include_sample, i],
        #         r0[include_sample, i],
        #         method,
        #     )
        #     pValues[method][node_names[i - 1]] = (test_res[1]).item()
        #     if method == "dr":
        #         print(node_names[i - 1], method, test_res)
        z1, z2 = (
            a0[:, i] - y,
            r0[:, i] - y,
        )
        mean_and_std(z1)
        mean_and_std(z2)
        for method in methods:
            test_res = ttest(
                y,
                a0[:, 0],
                r0[:, 0],
                a0[:, i],
                r0[:, i],
                method,
            )
            pValues[method][node_names[i - 1]] = (test_res[1]).item()
            if method == "dr":
                print(node_names[i - 1], method, test_res)
    print("______Corrected p values______")
    for method in methods:
        pValues[method].iloc[0, :] = sm.stats.multipletests(
            pValues[method].iloc[0, :], alpha=1 - confidence_level, method="fdr_by"
        )[1]
        causal_graph[method] = pValues[method].applymap(
            lambda p: int(p < (1 - confidence_level))
        )
    df = (
        pd.DataFrame(groundtruth)
        .rename(columns={"Y": "groundtruth"})
        .join((causal_graph["dr"]).T.rename(columns={0: "dr_causal_graph"}))
        .join((pValues["dr"]).T.rename(columns={0: "dr_p_values"}))
        .join((causal_graph["direct"]).T.rename(columns={0: "direct_causal_graph"}))
        .join((pValues["direct"]).T.rename(columns={0: "direct_p_values"}))
    )
    node_names_dict = dict(zip(df.index, var_names))
    df = df.rename(index=node_names_dict)
    # groundtruth = groundtruth.rename(index=node_names_dict)
    print(df.to_string())
    acc = calculate_accuracy(causal_graph["dr"], groundtruth)
    false_positives, false_negatives = sum(
        ((causal_graph["dr"]).T)[0][groundtruth == 0]
    ), -sum(((causal_graph["dr"]).T)[0][groundtruth == 1] - 1)
    gdt_positives, gdt_negatives = sum(groundtruth == 1), sum(groundtruth == 0)

    print(
        f'N_features: {experiment_config["N_feat"]}, N_data: {experiment_config["N_obs"]}'
    )
    print("accuracy: ", acc)
    print(
        "groundtruth positives: ", gdt_positives, "false negatives: ", false_negatives
    )
    print(
        "groundtruth negatives: ", gdt_negatives, "false positives: ", false_positives
    )

    if use_wandb:
        wandb.log({"groundtruth_positives": gdt_positives})
        wandb.log({"groundtruth_negatives": gdt_negatives})
        wandb.log({"false_negatives": false_negatives})
        wandb.log({"false_positives": false_positives})
        run.finish()

    return df["dr_causal_graph"]


def analyse_path(path):
    data_conf = {}

    if "linear" in path:
        data_conf["dataset_transform_type"] = "linear"
    elif "logsumexp" in path:
        data_conf["dataset_transform_type"] = "logsumexp"
    elif "sqrt" in path:
        data_conf["dataset_transform_type"] = "sqrt"
    else:
        data_conf["dataset_transform_type"] = "MLP"

    if "_pc_" in path:
        (_, a, _, pc, _, snr) = path.split("/")[-1].split("_")
        data_conf["data_coef_a"] = float(a)
        data_conf["data_pc"] = float(pc)
        data_conf["data_nsr"] = float(snr)

    if "real_data" in path:
        data_conf["dataset_type"] = "real_data"
        data_conf["N_feat"] = 25
        path = path.partition("real_data")[0] + "real_data"
    elif "time_series_data" in path:
        data_conf["dataset_type"] = "time_series_data"
        (N_feat, lag) = path.split("/")[-2].split("_")
        data_conf["N_feat"] = int(N_feat)
        data_conf["lag"] = int(lag)
    elif "synthetic_data" in path:
        data_conf["dataset_type"] = "synthetic_data"
        data_conf["N_feat"] = int(path.split("/")[-2])
    elif "independent_covariates_data" in path:
        data_conf["dataset_type"] = "independent_covariates_data"
        data_conf["N_feat"] = int(path.split("/")[-2])
        path = (
            path.partition("independent_covariates_data")[0]
            + "independent_covariates_data"
        )
    else:
        raise Exception("No other dataset type implemented")

    return data_conf, path


def compute_target(data_conf, parents_y):
    if data_conf["dataset_transform_type"] == "linear":
        y = data_conf["data_coef_a"] * parents_y.sum(axis=1)
    elif data_conf["dataset_transform_type"] == "logsumexp":
        y = data_conf["data_coef_a"] * torch.log(torch.exp(parents_y).sum(axis=1))
    elif data_conf["dataset_transform_type"] == "sqrt":
        y = data_conf["data_coef_a"] * torch.sqrt(parents_y.abs().sum(axis=1))
    elif data_conf["dataset_transform_type"] == "MLP":
        net = MLP(parents_y.shape[1]).eval()
        with torch.no_grad():
            y = data_conf["data_coef_a"] * net(parents_y)
    else:
        raise Exception("No other dataset transform implemented")
    y = y.reshape(-1, 1)
    y += math.sqrt(data_conf["data_nsr"]) * y.std() * torch.randn_like(y)
    return y


def get_data(path, N_data=-1):
    data_conf, path = analyse_path(path)
    if data_conf["dataset_type"] == "real_data":
        data_path = os.path.join(path, "data.csv")
        data = pd.read_csv(data_path, sep=",")
        data = data[(data.to_numpy() < 10).all(axis=1)]
        node_names = list(data.columns)
        groundtruth = pd.Series(
            np.array(
                [
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    1,
                    0,
                    1,
                    0,
                ]
            ),
            index=node_names,
            dtype=bool,
        )
        # groundtruth=pd.Series(np.random.rand(25)<data_conf['data_pc'],index=node_names,dtype=bool)
        X = data.to_numpy()[:N_data]
        _, _, X = standard_normalization(X)
        parents_y = X[:, groundtruth.to_numpy()]
        T_ones = np.ones((len(X), 1))
        X = torch.tensor(np.hstack((T_ones, X)), dtype=torch.float32)
        y = compute_target(data_conf, torch.tensor(parents_y, dtype=torch.float32))
        groundtruth = groundtruth.astype(int)
    elif data_conf["dataset_type"] == "time_series_data":
        X, y, groundtruth = create_dataset(
            1,
            N_data + 10 if N_data != -1 else 10000,
            data_conf["N_feat"],
            data_conf["lag"],
            data_conf["data_pc"],
            data_conf["dataset_transform_type"],
            data_conf["data_coef_a"],
            data_conf["data_nsr"],
        )
        N_seq = X.shape[0]
        N_col = X.shape[2]
        lag = groundtruth.shape[0] - 1
        Xs, ys = [], []
        for i in range(N_seq):
            cur_series = X[i]
            cur_target = y[i, :, 0]
            for j in range(lag, len(cur_series)):
                Xs.append(cur_series[j - lag : j].reshape(-1))
                ys.append(cur_target[j])
        X = torch.stack(Xs)
        T_ones = torch.ones((len(X), 1))
        X = torch.cat((T_ones, X), dim=1)
        X, y = X.float(), torch.tensor(ys).reshape(-1, 1).float()
        node_names = list(
            "X_" + str(i) + "_" + str(j) for i in range(-lag, 0) for j in range(N_col)
        )
        groundtruth = pd.Series(
            np.flip(groundtruth[1:, :-1, -1], axis=0).reshape(-1), index=node_names
        )
    elif data_conf["dataset_type"] == "synthetic_data":
        data_path = os.path.join(path, "simulated_data{}.csv".format(0))
        data = pd.read_csv(data_path, sep="\t", index_col=0)
        X = data.iloc[:, :-1].to_numpy()
        T_ones = np.ones((len(X), 1))
        X = np.hstack((T_ones, X))
        y = data.iloc[:, -1].to_numpy()
        node_names = list(data.columns[:-1])
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        ).reshape(-1, 1)
        groundtruth_path = os.path.join(path, "all_graph_matrix{}.csv".format(0))
        groundtruth = pd.read_csv(groundtruth_path, sep="\t", index_col=0)
        groundtruth = groundtruth.iloc[1:, 0]
        groundtruth = (groundtruth != 0).astype(int)[node_names]
    elif data_conf["dataset_type"] == "independent_covariates_data":
        node_names = ["X" + str(i + 1) for i in range(data_conf["N_feat"])]
        groundtruth = pd.Series(
            np.random.rand(data_conf["N_feat"]) < data_conf["data_pc"],
            index=node_names,
            dtype=bool,
        )
        # torch.manual_seed(0)
        X = torch.randn(N_data if N_data != -1 else 10000, data_conf["N_feat"]).float()
        parents_y = X[:, torch.tensor(groundtruth.to_numpy())]
        X = torch.cat([torch.ones((len(X), 1)), X], dim=1)
        y = compute_target(data_conf, parents_y)
        groundtruth = groundtruth.astype(int)
    else:
        raise Exception("No other dataset type implemented")

    if N_data != -1:
        X, y = X[:N_data], y[:N_data]
    data_conf["N_obs"] = len(X)
    return X, y, groundtruth, data_conf


def get_data_new(path: str, N_data=-1):
    data_conf = {}
    data = pd.read_csv(path)
    node_names = list(data.columns)
    X = data.iloc[:, :-1].to_numpy()
    T_ones = np.ones((len(X), 1))
    X = np.hstack((T_ones, X))
    y = data.iloc[:, -1].to_numpy()
    X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32
    ).reshape(-1, 1)

    ground_truth = pd.Series(np.array([1, 1, 0]), index=node_names, dtype=bool)
    data_conf["N_obs"] = len(X)
    data_conf["dataset_type"] = "simulated_examples"
    data_conf["dataset_transform_type"] = "linear"
    data_conf["N_feat"] = 6
    return X, y, ground_truth, data_conf


# XXX this is the part that I am using
class DoublyRobustClassWrapper:
    def __init__(
        self,
        data: Data,
        groundtruth: pd.DataFrame,
        data_conf: Dict,
        topological_order: List,
        target: str,
    ):
        var_names = [var for var in topological_order if var != target]
        target_index = topological_order.index(target)
        self.node_names = var_names
        self.groundtruth = groundtruth
        X = data.samples[
            :, [i for i in range(data.samples.shape[1]) if i != target_index]
        ]
        T_ones = np.ones((len(X), 1))
        X = np.hstack((T_ones, X))
        y = data.samples[:, target_index]
        self.X, self.y = torch.tensor(X, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        ).reshape(-1, 1)
        self.intervention_nodes = torch.tensor(data.nodes, dtype=torch.bool)

        print(f"The shape of X in this case is {self.X.shape}")
        data_conf["N_obs"] = X.shape[0]
        data_conf["N_feat"] = X.shape[1] - 1  # removing the bias term
        self.data_conf = data_conf.copy()

    def infer_causal_parents(self):
        num_epochs, learning_rate, l1_reg, l2_reg, batch_size = 500, 0.01, 0, 0, 32
        train_double = True
        N_obs = self.data_conf["N_obs"]
        direct_parents = infer_causal_parents_new(
            self.X,
            self.y,
            self.intervention_nodes,
            self.groundtruth,
            self.data_conf,
            num_epochs,
            learning_rate,
            l1_reg,
            l2_reg,
            batch_size,
            train_double,
            N_obs,
            self.node_names,
            closed_form=False,
            use_wandb=False,
        )
        return direct_parents


model_type = "MLP"
num_epochs, learning_rate, l1_reg, l2_reg, batch_size = 500, 0.01, 0, 0, 32
train_double = True

print("cuda is available", torch.cuda.is_available())

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Double robustness experiments")
    # parser.add_argument("--N_data", default=1000, type=int)
    # parser.add_argument("--path", type=str)
    # args = vars(parser.parse_args())
    # N_data = args["N_data"]
    # path = args["path"]

    path = "/vol/bitbucket/jd123/causal_bayes_opt/data/test.csv"
    N_data = 200
    infer_causal_parents(
        path,
        num_epochs,
        learning_rate,
        l1_reg,
        l2_reg,
        batch_size,
        train_double,
        N_data,
        closed_form=False,
        use_wandb=False,
    )
