import os
import pickle
import re
from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg

base_path = "/Users/jeandurand/Documents/Masters Thesis/causal_bayes_opt/results/Erdos"


# Function to load all results of a given type
def load_results(base_path, regex):
    results = []
    for filename in os.listdir(base_path):
        if bool(re.match(regex, filename)):
            with open(os.path.join(base_path, filename), "rb") as file:
                results.append(pickle.load(file))

    return results


# Function to load all results of a given type
def load_results_regex(base_path, regex):
    results = []
    base_path, regex = regex.split("/")
    for filename in os.listdir(base_path):
        if bool(re.match(regex, filename)):
            with open(os.path.join(base_path, filename), "rb") as file:
                results.append(pickle.load(file))

    return results


# Function to compute statistics across runs
def aggregate_results(results, key):
    all_data = [res[key] for res in results]
    mean_data = np.mean(all_data, axis=0)
    std_dev_data = np.std(all_data, axis=0)
    return mean_data, std_dev_data


def plot_posterior_nobs(
    graph_type: str,
    experiment: str,
    n_obs_list: list = [50, 100, 200],
    used_interventions: bool = True,
    nonlinear: bool = False,
    save_file: bool = False,
    D: int = 1000,
    ax=None,
):
    if ax is None:
        ax = plt.gca()
    sns.set_theme(style="whitegrid")  # Use seaborn style for better aesthetics

    # Define color map
    inferno = cm.get_cmap("plasma")
    colors = inferno(np.linspace(0, 1, len(n_obs_list)))

    nonlinear_string = "nonlinear" if nonlinear else "linear"
    intervention_string = "_nint" if used_interventions else ""

    ax.set_ylim(0.2, 1.1)

    for i, n in enumerate(n_obs_list):
        experiment_pattern = (
            rf".*_{nonlinear_string}_{n}{intervention_string}_D_{D}(_p_1)?_noise"
        )
        all_results = load_results(
            base_path=f"results/posterior/{graph_type}", regex=experiment_pattern
        )

        mean, std = aggregate_results(all_results, experiment)
        x_values = range(len(mean))

        ax.fill_between(x_values, mean - std, mean + std, alpha=0.2, color=colors[i])
        ax.plot(x_values, mean, label=f"{n} observations", color=colors[i], linewidth=2)

    ax.set_title(
        f"Posterior Distribution for {graph_type} ({nonlinear_string.capitalize()})"
    )
    ax.set_xlabel("Iterations")  # Replace with appropriate label
    ax.set_ylabel(f"Expected {experiment}")

    # ax.legend(title="Number of Observations", loc="best")
    ax.grid(True, linestyle="--", alpha=0.6)

    # if save_file:
    #     plt.savefig(f"{graph_type}_posterior_plot.png", dpi=300, bbox_inches="tight")
    # else:
    #     plt.show()


def plot_posterior_p(
    graph_type: str,
    experiment: str,
    n_obs: int = 200,
    used_interventions: bool = True,
    nonlinear: bool = False,
    p_list: List[int] = [1, 2, 3],
    save_file: bool = False,
    D: int = 1000,
):
    sns.set_theme(style="whitegrid")  # Use seaborn style for better aesthetics

    # Define color map
    inferno = cm.get_cmap("plasma")
    colors = inferno(np.linspace(0, 1, len(p_list)))

    nonlinear_string = "nonlinear" if nonlinear else "linear"
    intervention_string = "_nint" if used_interventions else ""

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.ylim(0.2, 1.1)

    for i, p in enumerate(p_list):
        if p == 1:
            experiment_pattern = rf".*_{nonlinear_string}_{n_obs}{intervention_string}_D_{D}(_p_1)?_noise"
        else:
            experiment_pattern = (
                rf".*_{nonlinear_string}_{n_obs}{intervention_string}_D_{D}_p_{p}_noise"
            )

        all_results = load_results(
            base_path=f"results/posterior/{graph_type}", regex=experiment_pattern
        )

        mean, std = aggregate_results(all_results, experiment)
        x_values = range(len(mean))

        plt.fill_between(x_values, mean - std, mean + std, alpha=0.2, color=colors[i])
        plt.plot(x_values, mean, label=f"{p} degree", color=colors[i], linewidth=2)

    plt.title(
        f"Posterior Distribution for {graph_type} ({nonlinear_string.capitalize()})"
    )
    plt.xlabel("Iterations")  # Replace with appropriate label
    plt.ylabel(f"Expected {experiment}")

    plt.legend(title="Expected degree", loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_file:
        plt.savefig(f"{graph_type}_posterior_plot.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_posterior_obs_int(
    graph_type: str,
    experiment: str,
    n_obs: int = 200,
    used_interventions: bool = [True, False],
    p: int = 1,
    nonlinear: bool = False,
    save_file: bool = False,
    D: int = 1000,
):
    sns.set_theme(style="whitegrid")  # Use seaborn style for better aesthetics

    # Define color map
    inferno = cm.get_cmap("plasma")
    colors = inferno(np.linspace(0, 1, len(used_interventions)))

    nonlinear_string = "nonlinear" if nonlinear else "linear"
    intervention_string = "_nint" if used_interventions else ""

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.ylim(0.2, 1.1)

    for i, used_int in enumerate(used_interventions):
        intervention_string = "_nint" if used_int else ""
        experiment_pattern = (
            rf".*_{nonlinear_string}_{n_obs}{intervention_string}_D_{D}(_p_1)?_noise"
        )

        all_results = load_results(
            base_path=f"results/posterior/{graph_type}", regex=experiment_pattern
        )

        mean, std = aggregate_results(all_results, experiment)
        x_values = range(len(mean))

        plt.fill_between(x_values, mean - std, mean + std, alpha=0.2, color=colors[i])
        plt.plot(x_values, mean, label=f"{used_int}", color=colors[i], linewidth=2)

    plt.title(
        f"Posterior Distribution for {graph_type} ({nonlinear_string.capitalize()})"
    )
    plt.xlabel("Iterations")  # Replace with appropriate label
    plt.ylabel(f"Expected {experiment}")

    plt.legend(title="Interventional samples", loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_file:
        plt.savefig(f"{graph_type}_posterior_plot.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_posterior_obs_int(
    graph_type: str,
    experiment: str,
    n_obs: int = 200,
    used_interventions: bool = [True, False],
    p: int = 1,
    nonlinear: bool = False,
    save_file: bool = False,
    D: int = 1000,
):
    sns.set_theme(style="whitegrid")  # Use seaborn style for better aesthetics

    # Define color map
    inferno = cm.get_cmap("plasma")
    colors = inferno(np.linspace(0, 1, len(used_interventions)))

    nonlinear_string = "nonlinear" if nonlinear else "linear"
    intervention_string = "_nint" if used_interventions else ""

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.ylim(0.2, 1.1)

    for i, used_int in enumerate(used_interventions):
        intervention_string = "_nint" if used_int else ""
        experiment_pattern = (
            rf".*_{nonlinear_string}_{n_obs}{intervention_string}_D_{D}(_p_1)?_noise"
        )

        all_results = load_results(
            base_path=f"results/posterior/{graph_type}", regex=experiment_pattern
        )

        mean, std = aggregate_results(all_results, experiment)
        x_values = range(len(mean))

        plt.fill_between(x_values, mean - std, mean + std, alpha=0.2, color=colors[i])
        plt.plot(x_values, mean, label=f"{used_int}", color=colors[i], linewidth=2)

    plt.title(
        f"Posterior Distribution for {graph_type} ({nonlinear_string.capitalize()})"
    )
    plt.xlabel("Iterations")  # Replace with appropriate label
    plt.ylabel(f"Expected {experiment}")

    plt.legend(title="Interventional samples", loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_file:
        plt.savefig(f"{graph_type}_posterior_plot.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_posterior_noise(
    graph_type: str,
    experiment: str,
    n_obs: int = 200,
    used_interventions: bool = True,
    noise_list: List = [0.1, 1.0, 2.0, 3.0],
    nonlinear: bool = False,
    save_file: bool = False,
    D: int = 1000,
):
    sns.set_theme(style="whitegrid")  # Use seaborn style for better aesthetics

    # Define color map
    inferno = cm.get_cmap("plasma")
    colors = inferno(np.linspace(0, 1, len(noise_list)))

    nonlinear_string = "nonlinear" if nonlinear else "linear"
    intervention_string = "_nint" if used_interventions else ""

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.ylim(0.2, 1.1)

    for i, noise in enumerate(noise_list):
        experiment_pattern = rf".*_{nonlinear_string}_{n_obs}{intervention_string}_dr_D_{D}_p_1_noise_{noise}"
        all_results = load_results(
            base_path=f"results/posterior/{graph_type}", regex=experiment_pattern
        )
        print(experiment_pattern)
        mean, std = aggregate_results(all_results, experiment)
        x_values = range(len(mean))

        plt.fill_between(x_values, mean - std, mean + std, alpha=0.2, color=colors[i])
        plt.plot(x_values, mean, label=f"Noise {noise}", color=colors[i], linewidth=2)

    plt.title(
        f"Posterior Distribution for {graph_type} ({nonlinear_string.capitalize()})"
    )
    plt.xlabel("Iterations")  # Replace with appropriate label
    plt.ylabel(f"Expected {experiment}")

    plt.legend(title="Interventional samples", loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_file:
        plt.savefig(f"{graph_type}_posterior_plot.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_posterior_dr(
    graph_type: str,
    experiment: str,
    n_obs: int = 200,
    used_interventions: bool = True,
    nonlinear: bool = False,
    save_file: bool = False,
    D: int = 1000,
):
    pass


def plot_acquisitions(
    graph_type: str,
    experiment: str,
    n_obs: int = 200,
    acquisition_list: List = ["EI", "CEO", "PES"],
    use_dr: bool = False,
    save_file: bool = False,
):
    sns.set_theme(style="whitegrid")  # Use seaborn style for better aesthetics

    # Define color map
    inferno = cm.get_cmap("plasma")
    colors = inferno(np.linspace(0, 1, len(acquisition_list)))
    dr_string = "_dr" if use_dr else "_all"

    plt.figure(figsize=(10, 6))  # Set the figure size

    for i, acquisition in enumerate(acquisition_list):
        experiment_pattern = rf".*_cbo{dr_string}_{acquisition}_results_{n_obs}"
        all_results = load_results(
            base_path=f"results/{graph_type}", regex=experiment_pattern
        )

        mean, std = aggregate_results(all_results, experiment)
        x_values = range(len(mean))

        plt.fill_between(x_values, mean - std, mean + std, alpha=0.2, color=colors[i])
        plt.plot(x_values, mean, label=acquisition, color=colors[i], linewidth=2)

    plt.title(f"CBO Unknown for different acquisition functions")
    plt.xlabel("Iterations")  # Replace with appropriate label
    plt.ylabel(f"Expected {experiment}")

    plt.legend(title="Interventional samples", loc="best")
    plt.grid(True, linestyle="--", alpha=0.6)

    if save_file:
        plt.savefig(f"{graph_type}_acquisitions.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_everything(
    base_path,
    experiment,
    n_obs=200,
    n_int=2,
    nonlinear=False,
    save_file=False,
    plot_all=True,
    plot_dr1=False,
    plot_dr2=True,
    plot_cbo=True,
    plot_random=True,
):
    sns.set_theme(style="whitegrid")
    inferno = cm.get_cmap("plasma")
    nonlinear_string = "_nonlinear" if nonlinear else ""

    # Initialize plot for CEO results

    if plot_dr2:
        # Load and aggregate results for CEO
        dr2_string = (
            rf".*_cbo_unknown_dr2_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        dr2_results = load_results(base_path, dr2_string)
        dr2_mean, dr2_std = aggregate_results(dr2_results, experiment)
        x_values = range(len(dr2_mean))
        plt.fill_between(
            x_values, dr2_mean - dr2_std, dr2_mean + dr2_std, alpha=0.2, color="Red"
        )
        plt.plot(x_values, dr2_mean, label="CBO Unknown", color="Red")

    if plot_cbo:
        # Load and aggregate results for CEO
        cbo_string = rf".*_cbo_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results = load_results(base_path, cbo_string)
        cbo_mean, cbo_std = aggregate_results(cbo_results, experiment)
        cbo_mean = cbo_mean[1:]
        cbo_std = cbo_std[1:]
        x_values = range(len(cbo_mean))
        plt.fill_between(
            x_values, cbo_mean - cbo_std, cbo_mean + cbo_std, alpha=0.2, color="Black"
        )
        plt.plot(x_values, cbo_mean, label="CBO", color="Black")

    if plot_random:
        # Load and aggregate results for CEO
        random_string = (
            rf".*_cbo_results_random_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        random_results = load_results(base_path, random_string)
        random_mean, random_std = aggregate_results(random_results, experiment)
        x_values = range(len(random_mean))
        plt.fill_between(
            x_values,
            random_mean - random_std,
            random_mean + random_std,
            alpha=0.2,
            color="Green",
        )
        plt.plot(x_values, random_mean, label="Random", color="Green")

    # Final plot adjustments
    plt.xlabel("Trial")
    plt.ylabel("Y value")
    plt.grid(True)
    plt.legend()

    if save_file:
        filename = f"{base_path}/{experiment}_{n_obs}_{n_int}"
        plt.savefig(filename)

    plt.show()


def all_means_final_project(regex_list):
    method_map = {0: "CBO Unknown", 1: "CBO", 2: "CEO", 3: "BO", 4: "CBO wrong"}
    mean_results = {}
    for i, regex in enumerate(regex_list):
        results = load_results_regex("", regex=regex)
        results = np.hstack([result["Per_trial_Y"][:30] for result in results])
        cbo_mean = np.mean(results)
        cbo_std = np.std(results)
        mean_results[method_map[i]] = {"mean": cbo_mean, "std": cbo_std}

    return mean_results


def all_min_final_project(regex_list):
    method_map = {0: "CBO Unknown", 1: "CBO", 2: "CEO", 3: "BO", 4: "CBO wrong"}
    min_results = {}
    for i, regex in enumerate(regex_list):
        results = load_results_regex("", regex=regex)
        results = np.vstack([result["Best_Y"][:30] for result in results])
        best_results = results.min(axis=1)
        cbo_mean = np.mean(best_results)
        cbo_std = np.std(best_results)
        min_results[method_map[i]] = {"mean": cbo_mean, "std": cbo_std}

    return min_results


def all_means(
    base_path,
    n_obs=200,
    n_int=2,
    nonlinear: bool = False,
    has_all: bool = True,
    has_dr1: bool = True,
    has_dr2: bool = True,
    has_cbo: bool = True,
    has_random: bool = True,
):

    mean_results = {}
    nonlinear_string = "_nonlinear" if nonlinear else ""

    if has_cbo:
        cbo_string = rf".*_cbo_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results = load_results(base_path, cbo_string)
        cbo_results = np.hstack(
            [cbo_result["Per_trial_Y"] for cbo_result in cbo_results]
        )
        cbo_mean = np.mean(cbo_results)
        cbo_std = np.std(cbo_results)
        mean_results["cbo"] = {"mean": cbo_mean, "std": cbo_std}

    if has_dr2:
        dr2_string = (
            rf".*_cbo_unknown_dr2_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        dr2_results = load_results(base_path, dr2_string)
        dr2_results = np.hstack(
            [dr2_result["Per_trial_Y"] for dr2_result in dr2_results]
        )
        dr2_mean = np.mean(dr2_results)
        dr2_std = np.std(dr2_results)
        mean_results["dr2"] = {"mean": dr2_mean, "std": dr2_std}

    if has_random:
        random_string = (
            rf".*_cbo_results_random_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        random_results = load_results(base_path, random_string)
        random_results = np.hstack(
            [random_result["Per_trial_Y"] for random_result in random_results]
        )
        random_mean = np.mean(random_results)
        random_std = np.std(random_results)
        mean_results["random"] = {"mean": random_mean, "std": random_std}

    return mean_results


def all_best(
    base_path,
    n_obs=200,
    n_int=2,
    nonlinear: bool = False,
    has_all: bool = True,
    has_dr1: bool = True,
    has_dr2: bool = True,
    has_cbo: bool = True,
    has_random: bool = True,
):

    min_results = {}
    nonlinear_string = "_nonlinear" if nonlinear else ""

    if has_cbo:
        cbo_string = rf".*_cbo_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results = load_results(base_path, cbo_string)
        cbo_results = np.vstack([cbo_result["Best_Y"] for cbo_result in cbo_results])
        cbo_best_results = cbo_results.min(axis=1)
        cbo_mean = np.mean(cbo_best_results)
        cbo_std = np.std(cbo_best_results)
        min_results["cbo"] = {"mean": cbo_mean, "std": cbo_std}

    if has_dr2:
        dr2_string = (
            rf".*_cbo_unknown_dr2_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        dr2_results = load_results(base_path, dr2_string)
        dr2_results = np.vstack([dr2_result["Best_Y"] for dr2_result in dr2_results])
        dr2_best_results = dr2_results.min(axis=1)
        dr2_mean = np.mean(dr2_best_results)
        dr2_std = np.std(dr2_best_results)
        min_results["dr2"] = {"mean": dr2_mean, "std": dr2_std}

    if has_random:
        random_string = (
            rf".*_cbo_results_random_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        random_results = load_results(base_path, random_string)
        random_results = np.vstack(
            [random_result["Best_Y"] for random_result in random_results]
        )
        random_best_results = random_results.min(axis=1)
        random_mean = np.mean(random_best_results)
        random_std = np.std(random_best_results)
        min_results["random"] = {"mean": random_mean, "std": random_std}

    return min_results


def iterations_to_min(
    base_path, num_cbo_graphs, n_obs=100, n_int=2, noiseless=True, graph_idxs=None
):
    argmin_results = {}
    noisy_suffix = r"\.pickle" if noiseless else r"noisy\.pickle"
    noisy_string = "" if noiseless else "_noisy"

    # Load and aggregate results for CEO
    ceo_string = rf".*_ceo_.*_results_{n_obs}_{n_int}_{noisy_suffix}"
    ceo_results = load_results(base_path, ceo_string)
    ceo_results = np.vstack([ceo_result["Best_Y"] for ceo_result in ceo_results])
    ceo_best_results = ceo_results.argmin(axis=1)
    ceo_mean = np.mean(ceo_best_results)
    ceo_std = np.std(ceo_best_results)
    argmin_results["ceo"] = {"mean": ceo_mean, "std": ceo_std}

    # Load and aggregate results for BO
    # Load and aggregate results for CEO
    bo_string = rf".*_bo_results_{n_obs}_{n_int}_{noisy_suffix}"
    bo_results = load_results(base_path, bo_string)
    bo_results = np.vstack([bo_result["Best_Y"] for bo_result in bo_results])
    bo_best_results = bo_results.argmin(axis=1)
    bo_mean = np.mean(bo_best_results)
    bo_std = np.std(bo_best_results)
    argmin_results["bo"] = {"mean": bo_mean, "std": bo_std}

    for graph_index in range(num_cbo_graphs):
        cbo_string = (
            rf".*_cbo_results_{n_obs}_{n_int}_graph_{graph_index}{noisy_suffix}"
        )
        cbo_results = load_results(base_path, cbo_string)
        cbo_results = np.vstack([cbo_result["Best_Y"] for cbo_result in cbo_results])
        cbo_best_results = cbo_results.argmin(axis=1)
        cbo_mean = np.mean(cbo_best_results)
        cbo_std = np.std(cbo_best_results)
        argmin_results[f"cbo_{graph_index}"] = {"mean": cbo_mean, "std": cbo_std}

    return argmin_results


def summarise_n_observational(statistic="accuracy"):
    sns.set_theme(style="whitegrid")
    graphs = [5, 6, 7, 8, 9, 10]
    n_obs_list = [50, 100, 200]
    used_interventions = True
    intervention_string = "_nint" if used_interventions else ""

    # Define colors: blue for the first, orange for the second, and black for the third
    colors = ["blue", "orange", "black"]

    fig, axs = plt.subplots(
        2, 1, figsize=(6, 10)
    )  # Create a figure with 2 subplots side by side

    # Left plot with nonlinear experiment pattern
    for i, n in enumerate(n_obs_list):
        f1_means = []
        f1_stds = []

        for graph in graphs:
            experiment_pattern = (
                rf".*_nonlinear_{n}{intervention_string}_D_1000_p_1_noise_"
            )
            all_results = load_results(
                base_path=f"posterior/Erdos{graph}/", regex=experiment_pattern
            )
            final_f1_scores = [result[statistic][-1] for result in all_results]
            f1_means.append(np.mean(final_f1_scores))
            f1_stds.append(np.std(final_f1_scores))

        axs[1].errorbar(
            graphs,
            f1_means,
            yerr=f1_stds,
            capsize=5,
            label=f"N={n}",
            marker="o",
            color=colors[i],
        )

    axs[1].set_xlabel("Graphs")
    axs[1].set_ylabel(f"Mean {statistic}")
    axs[1].set_title("Nonlinear Graphs")
    axs[1].set_xticks(graphs)
    axs[1].grid(True)

    # Right plot with linear experiment pattern
    for i, n in enumerate(n_obs_list):
        f1_means = []
        f1_stds = []

        for graph in graphs:
            experiment_pattern = (
                rf".*_linear_{n}{intervention_string}_D_1000_p_1_noise_"
            )
            all_results = load_results(
                base_path=f"posterior/Erdos{graph}/", regex=experiment_pattern
            )
            final_f1_scores = [result[statistic][-1] for result in all_results]
            f1_means.append(np.mean(final_f1_scores))
            f1_stds.append(np.std(final_f1_scores))

        axs[0].errorbar(
            graphs,
            f1_means,
            yerr=f1_stds,
            capsize=5,
            label=f"N={n}",
            marker="o",
            color=colors[i],
        )

    # Set the same limits for both y-axes
    axs[0].set_ylim(0.2, 1.1)
    axs[1].set_ylim(0.2, 1.1)
    axs[0].set_xlabel("Graphs")
    axs[0].set_ylabel(f"Mean {statistic}")
    axs[0].set_title("Linear Graphs")
    axs[0].set_xticks(graphs)
    axs[0].grid(True)

    # Add a single legend at the bottom
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(n_obs_list),
        title="Observational Dataset Size",
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at the bottom for the legend
    plt.savefig(f"observational_size_results_{statistic}")
    plt.show()


def summarise_p_density(statistic="accuracy"):
    sns.set_theme(style="whitegrid")
    graphs = [5, 6, 7, 8, 9, 10]
    p_list = [1, 2, 3]
    used_interventions = True
    intervention_string = "_nint" if used_interventions else ""

    # Define colors: blue for P=1, orange for P=2, and black for P=3
    colors = ["blue", "orange", "black"]

    fig, axs = plt.subplots(
        2, 1, figsize=(6, 10)
    )  # Create a figure with 2 subplots side by side

    # Left plot with nonlinear experiment pattern
    for i, p in enumerate(p_list):
        f1_means = []
        f1_stds = []

        for graph in graphs:
            experiment_pattern = (
                rf".*_nonlinear_200{intervention_string}_D_1000_p_{p}_noise_"
            )
            all_results = load_results(
                base_path=f"posterior/Erdos{graph}/", regex=experiment_pattern
            )
            final_f1_scores = [result[statistic][-1] for result in all_results]
            f1_means.append(np.mean(final_f1_scores))
            f1_stds.append(np.std(final_f1_scores))

        axs[1].errorbar(
            graphs,
            f1_means,
            yerr=f1_stds,
            capsize=5,
            label=f"P={p}",
            marker="o",
            color=colors[i],
        )

    axs[1].set_xlabel("Graphs")
    axs[1].set_ylabel(f"Mean {statistic}")
    axs[1].set_title("Nonlinear Graphs")
    axs[1].set_xticks(graphs)
    # axs[1].legend(title="Graph Density")
    axs[1].grid(True)

    # Right plot with linear experiment pattern
    for i, p in enumerate(p_list):
        f1_means = []
        f1_stds = []

        for graph in graphs:
            experiment_pattern = (
                rf".*_linear_200{intervention_string}_D_1000_p_{p}_noise_"
            )
            all_results = load_results(
                base_path=f"posterior/Erdos{graph}/", regex=experiment_pattern
            )
            final_f1_scores = [result[statistic][-1] for result in all_results]
            f1_means.append(np.mean(final_f1_scores))
            f1_stds.append(np.std(final_f1_scores))

        axs[0].errorbar(
            graphs,
            f1_means,
            yerr=f1_stds,
            capsize=5,
            label=f"P={p}",
            marker="o",
            color=colors[i],
        )

    # Set the same limits for both y-axes
    axs[0].set_ylim(0, 1.1)
    axs[1].set_ylim(0, 1.1)
    axs[0].set_xlabel("Graphs")
    axs[0].set_ylabel(f"Mean {statistic}")
    axs[0].set_title("Linear Graphs")
    axs[0].set_xticks(graphs)
    # axs[0].legend(title="Graph Density")
    axs[0].grid(True)

    # Add a single legend at the bottom
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(p_list),
        title="Graph Density",
    )

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust spacing to prevent overlap
    plt.savefig(f"graph_density_results_{statistic}")
    plt.show()


def summarise_cbo_erdos_results(graphs: List = ["Erdos10", "Erdos15", "Erdos20"]):
    sns.set_theme(style="whitegrid")
    mean_results = {}
    min_results = {}

    # Extract linear results
    for graph in graphs:
        mean_results[graph] = all_means(graph, nonlinear=False)
        min_results[graph] = all_best(graph, nonlinear=False)

    # Function to extract data for plotting
    def extract_plot_data(data):
        methods = ["cbo", "dr2", "random"]
        means = []
        stds = []

        for method in methods:
            means.append(
                [
                    data["Erdos10"][method]["mean"],
                    data["Erdos15"][method]["mean"],
                    data["Erdos20"][method]["mean"],
                ]
            )
            stds.append(
                [
                    data["Erdos10"][method]["std"],
                    data["Erdos15"][method]["std"],
                    data["Erdos20"][method]["std"],
                ]
            )

        return methods, np.array(means).T, np.array(stds).T

    # Prepare plotting
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    # axs.tick_params(axis="both", labelsize=14)

    experiments = ["Erdos10", "Erdos15", "Erdos20"]

    # Plot linear results on the top row
    methods, means1, stds1 = extract_plot_data(mean_results)
    _, means2, stds2 = extract_plot_data(min_results)

    methods = ["cbo", "cbo unknown", "random"]
    for i, experiment in enumerate(experiments):
        ax = axs[0, i]
        ax.tick_params(axis="both", labelsize=14)
        ax.errorbar(
            methods,
            means1[i],
            yerr=stds1[i],
            label="Average",
            marker="o",
            capsize=5,
            linestyle="-",
            color="blue",
        )
        ax.errorbar(
            methods,
            means2[i],
            yerr=stds2[i],
            label="Best",
            marker="o",
            capsize=5,
            linestyle="-",
            color="orange",
        )
        ax.set_title(f"{experiment} (Linear)", fontsize=18)
        ax.set_ylabel("Mean Values", fontsize=16)

    # Extract nonlinear results
    for graph in graphs:
        mean_results[graph] = all_means(graph, nonlinear=True)
        min_results[graph] = all_best(graph, nonlinear=True)

    # Plot nonlinear results on the bottom row
    methods, means1, stds1 = extract_plot_data(mean_results)
    _, means2, stds2 = extract_plot_data(min_results)

    methods = ["cbo", "cbo unknown", "random"]
    for i, experiment in enumerate(experiments):
        ax = axs[1, i]
        ax.tick_params(axis="both", labelsize=14)
        ax.errorbar(
            methods,
            means1[i],
            yerr=stds1[i],
            label="Average",
            marker="o",
            capsize=5,
            linestyle="-",
            color="blue",
        )
        ax.errorbar(
            methods,
            means2[i],
            yerr=stds2[i],
            label="Best",
            marker="o",
            capsize=5,
            linestyle="-",
            color="orange",
        )
        ax.set_title(f"{experiment} (Nonlinear)", fontsize=18)
        ax.set_ylabel("Mean Values", fontsize=16)

    # Add a single legend at the bottom
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("erdos_summary_cbo.png")
    plt.show()


def summarise_erdos20_per_iteration():
    # First plot
    plot_everything(
        base_path="Erdos20",
        experiment="Best_Y",
        nonlinear=True,
        n_obs=200,
        plot_cbo=True,
        plot_all=False,
        plot_dr2=True,
        plot_random=True,
    )
    fig1 = plt.gcf()
    fig1.canvas.draw()
    img1 = np.array(fig1.canvas.renderer._renderer)
    plt.close(fig1)

    # Second plot
    plot_everything(
        base_path="Erdos20",
        experiment="Per_trial_Y",
        nonlinear=True,
        n_obs=200,
        plot_cbo=True,
        plot_all=False,
        plot_dr2=True,
        plot_random=True,
    )
    fig2 = plt.gcf()
    fig2.canvas.draw()
    img2 = np.array(fig2.canvas.renderer._renderer)
    plt.close(fig2)

    # Create a new composite figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))

    # Add the captured plots to the new figure
    ax1.imshow(img1)
    ax1.axis("off")
    ax1.set_title(
        "Erdos 20 (Nonlinear) - Best Y", fontsize=18, pad=-10
    )  # Decrease pad to move title closer

    ax2.imshow(img2)
    ax2.axis("off")
    ax2.set_title(
        "Erdos 20 (Nonlinear) - Average Y", fontsize=18, pad=-10
    )  # Decrease pad to move title closer

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust top margin
    plt.savefig("erdos20_nonlinear_cbo", bbox_inches="tight")
    plt.show()


def summarise_ecoli1_per_iteration():
    # First plot
    plot_everything(
        base_path="Ecoli1",
        experiment="Best_Y",
        # nonlinear=True,
        n_obs=200,
        n_int=1,
        plot_cbo=True,
        plot_all=False,
        plot_dr2=True,
        plot_random=True,
    )
    fig1 = plt.gcf()
    fig1.canvas.draw()
    img1 = np.array(fig1.canvas.renderer._renderer)
    plt.close(fig1)

    # Second plot
    plot_everything(
        base_path="Ecoli1",
        experiment="Per_trial_Y",
        # nonlinear=True,
        n_obs=200,
        n_int=1,
        plot_cbo=True,
        plot_all=False,
        plot_dr2=True,
        plot_random=True,
    )
    fig2 = plt.gcf()
    fig2.canvas.draw()
    img2 = np.array(fig2.canvas.renderer._renderer)
    plt.close(fig2)

    # Create a new composite figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # Add the captured plots to the new figure
    ax1.imshow(img1)
    ax1.axis("off")
    ax1.set_title(
        "E. Coli - Best Y", fontsize=18, pad=-10
    )  # Decrease pad to move title closer

    ax2.imshow(img2)
    ax2.axis("off")
    ax2.set_title(
        "E. Coli - Average Y", fontsize=18, pad=-10
    )  # Decrease pad to move title closer

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust top margin
    plt.savefig("ecoli1_cbo", bbox_inches="tight")
    plt.show()


def all_uncertainties(
    base_path, num_cbo_graphs, n_obs=100, n_int=2, noiseless=True, graph_idxs=None
):
    uncertanties = {}
    noisy_suffix = r"\.pickle" if noiseless else r"noisy\.pickle"
    noisy_string = "" if noiseless else "_noisy"

    # Load and aggregate results for BO
    # Load and aggregate results for CEO
    bo_string = rf".*_bo_uncertainties_{n_obs}_{n_int}_{noisy_suffix}"
    bo_results = load_results(base_path, bo_string)
    bo_results = np.vstack([bo_result["total"] for bo_result in bo_results])
    bo_mean = np.mean(bo_results)
    bo_std = np.std(bo_results)
    uncertanties["bo"] = {"mean": bo_mean, "std": bo_std}

    for graph_index in range(num_cbo_graphs):
        cbo_string = (
            rf".*_cbo_uncertainties_{n_obs}_{n_int}_graph_{graph_index}{noisy_suffix}"
        )
        cbo_results = load_results(base_path, cbo_string)
        cbo_results = np.vstack([cbo_result["total"] for cbo_result in cbo_results])
        cbo_mean = np.mean(cbo_results)
        cbo_std = np.std(cbo_results)
        uncertanties[f"cbo_{graph_index}"] = {"mean": cbo_mean, "std": cbo_std}

    return uncertanties


def box_plots_mean(mean_results: dict, experiment: str):
    # Setup the plot
    fig, ax = plt.subplots()
    for i, (key, val) in enumerate(mean_results.items()):
        mean, std = val["mean"], val["std"]
        # Create the box
        box = plt.Rectangle(
            (i - 0.4, mean - std), 0.8, 2 * std, color="lightblue", alpha=0.5
        )
        ax.add_patch(box)
        # Plot the mean line
        plt.plot([i - 0.4, i + 0.4], [mean, mean], color="red")

    # Set plot properties
    plt.xticks(range(len(mean_results)), labels=mean_results.keys(), rotation=45)
    plt.ylabel("Value")
    plt.title(f"{experiment} y value for each algorithm")
    plt.grid(True)
    plt.show()


def plot_acquisitions_all(
    graphs: List = ["ToyGraphUnknown", "Graph5Unknown", "Graph6Unknown"],
    n_obs: int = 200,
    acquisition_list: List = ["EI", "CEO", "PES"],
    use_dr_list: List[bool] = [True, False],
    save_file: bool = False,
):
    sns.set_theme(style="whitegrid")  # Use seaborn style for better aesthetics

    # Define colors for DR (red) and No DR (black)
    colors = ["blue", "orange"]

    # Create a figure with 2 rows and len(graphs) columns (1 for each graph)
    fig, axs = plt.subplots(2, len(graphs), figsize=(18, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for col_idx, graph in enumerate(graphs):
        for row_idx, metric in enumerate(["Best_Y", "Per_trial_Y"]):
            ax = axs[row_idx, col_idx]

            for i, (use_dr, color) in enumerate(zip(use_dr_list, colors)):
                means = []
                stds = []
                for acquisition in acquisition_list:
                    dr_string = "_dr" if use_dr else "_all"
                    experiment_pattern = (
                        rf".*_cbo{dr_string}_{acquisition}_results_{n_obs}"
                    )
                    all_results = load_results(
                        base_path=f"results/{graph}", regex=experiment_pattern
                    )

                    if metric == "Best_Y":
                        res = np.vstack([result[metric][:30] for result in all_results])
                        res = res.min(axis=1)
                    else:
                        res = np.hstack([result[metric][:30] for result in all_results])

                    mean_res = np.mean(res)
                    std_res = np.std(res)

                    means.append(mean_res)
                    stds.append(std_res)

                # Plot with error bars
                ax.errorbar(
                    acquisition_list,
                    means,
                    yerr=stds,
                    label=f"{'DR' if use_dr else 'No DR'}",
                    linestyle="-",
                    marker="o",
                    color=color,
                    capsize=5,
                )

            ax.set_title(f"{graph}")
            if row_idx == 0:
                ax.set_ylabel("Best Y")
            else:
                ax.set_ylabel("Average Y")
            ax.set_xlabel("Acquisition Function")

            # Apply a slight margin to the y-axis and x-axis
            ax.margins(x=0.2, y=0.2)

    # Add a single legend at the bottom
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(use_dr_list))

    # Adjust layout for better visualization
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at the bottom for the legend

    plt.savefig("acquisition_comparison.png")
    plt.show()


def plot_acquisitions_all_boxplot(
    graphs: List = ["ToyGraphUnknown", "Graph5Unknown", "Graph6Unknown"],
    n_obs: int = 200,
    acquisition_list: List = ["EI", "CEO", "PES"],
    use_dr_list: List[bool] = [True, False],
    save_file: bool = False,
):
    sns.set_theme(style="whitegrid")  # Use seaborn style for better aesthetics

    # Define colors for DR (blue) and No DR (orange)
    colors = ["blue", "orange"]

    # Create a figure with 2 rows and len(graphs) columns (1 for each graph)
    fig, axs = plt.subplots(2, len(graphs), figsize=(18, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    for col_idx, graph in enumerate(graphs):
        for row_idx, metric in enumerate(["Best_Y", "Per_trial_Y"]):
            ax = axs[row_idx, col_idx]

            plot_data = {"Acquisition": [], "Value": [], "Type": []}

            for use_dr, color in zip(use_dr_list, colors):
                for acquisition in acquisition_list:
                    dr_string = "_dr" if use_dr else "_all"
                    experiment_pattern = (
                        rf".*_cbo{dr_string}_{acquisition}_results_{n_obs}"
                    )
                    all_results = load_results(
                        base_path=f"results/{graph}", regex=experiment_pattern
                    )

                    if metric == "Best_Y":
                        res = np.vstack([result[metric][:30] for result in all_results])
                        res = res.min(axis=1)
                    else:
                        res = np.hstack([result[metric][:30] for result in all_results])

                    # Add data to plot_data dictionary
                    plot_data["Acquisition"].extend([acquisition] * len(res))
                    plot_data["Value"].extend(res)
                    plot_data["Type"].extend(["DR" if use_dr else "No DR"] * len(res))

            # Create boxplots
            sns.boxplot(
                ax=ax,
                x="Acquisition",
                y="Value",
                hue="Type",
                data=plot_data,
                showfliers=False,
                palette=colors,
            )

            # Remove the legend for individual plots
            ax.get_legend().remove()

            # Set title and labels
            ax.set_title(f"{graph}")
            ax.set_ylabel("Best Y" if row_idx == 0 else "Average Y")
            ax.set_xlabel("Acquisition Function")
            # ax.margins(y=0.4)

            # Set specific y-axis limits for each graph
            if graph == "ToyGraphUnknown":
                ax.set_ylim(-2.5, 1)
            elif graph == "Graph5Unknown" and row_idx == 0:
                ax.set_ylim(-2, -0.5)
            elif graph == "Graph5Unknown" and row_idx == 1:
                ax.set_ylim(-2, 3)
            elif graph == "Graph6Unknown":
                ax.set_ylim(12.8, 14.4)

    # Add a single legend at the bottom
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=len(use_dr_list),
    )

    # Adjust layout for better visualization
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at the bottom for the legend

    if save_file:
        plt.savefig("acquisition_comparison.png", bbox_inches="tight")

    plt.show()


def summarise_cbo_toy_results():
    selected_graphs6 = [
        rf"Graph6Unknown/.*_cbo_unknown_results_200_2",
        rf"Graph6/.*_cbo_results_200_2_graph_0",
        rf"Graph6/.*ceo_\d*_results_200_2_",
        rf"Graph6/.*_bo_results_new_200_2_",
        rf"Graph6/.*_cbo_results_200_2_graph_4",
    ]
    selected_graphs5 = [
        rf"Graph5Unknown/.*_cbo_unknown_results_200_2",
        rf"Graph5/.*_cbo_results_200_2_graph_0",
        rf"Graph5/.*ceo_\d*_results_200_2_",
        rf"Graph5Wrong/.*_bo_results_new_200_2_",
        rf"Graph5Wrong/.*_cbo_results_200_2_graph_0",
    ]
    selected_toy = [
        rf"ToyGraphUnknown/.*_cbo_unknown_results_200_2",
        rf"ToyGraph/.*_cbo_results_200_2_graph_0",
        rf"ToyGraph/.*ceo_\d*_results_200_2_",
        rf"ToyGraph/.*_bo_results_new_200_2_",
        rf"ToyGraph/.*_cbo_results_200_2_graph_1",
    ]
    sns.set_theme(style="whitegrid")
    mean_results = {}
    min_results = {}
    mean_results["ToyGraph"] = all_means_final_project(selected_toy)
    mean_results["Graph5"] = all_means_final_project(selected_graphs5)
    mean_results["Graph6"] = all_means_final_project(selected_graphs6)

    min_results["ToyGraph"] = all_min_final_project(selected_toy)
    min_results["Graph5"] = all_min_final_project(selected_graphs5)
    min_results["Graph6"] = all_min_final_project(selected_graphs6)

    # Function to extract data for plotting
    def extract_plot_data(data):
        methods = list(data["ToyGraph"].keys())
        means = []
        stds = []

        for method in methods:
            means.append(
                [
                    data["ToyGraph"][method]["mean"],
                    data["Graph5"][method]["mean"],
                    data["Graph6"][method]["mean"],
                ]
            )
            stds.append(
                [
                    data["ToyGraph"][method]["std"],
                    data["Graph5"][method]["std"],
                    data["Graph6"][method]["std"],
                ]
            )

        return methods, np.array(means).T, np.array(stds).T

    # Prepare plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    experiments = ["ToyGraph", "Epidemiology", "Healthcare"]

    # Plot linear results on the top row
    methods, means1, stds1 = extract_plot_data(mean_results)
    _, means2, stds2 = extract_plot_data(min_results)

    for i, experiment in enumerate(experiments):
        ax = axs[i]
        ax.tick_params(axis="both", labelsize=14)
        ax.errorbar(
            methods,
            means1[i],
            yerr=stds1[i],
            label="Average",
            marker="o",
            capsize=5,
            linestyle="-",
            color="blue",
        )
        ax.errorbar(
            methods,
            means2[i],
            yerr=stds2[i],
            label="Best",
            marker="o",
            capsize=5,
            linestyle="-",
            color="orange",
        )
        ax.set_title(f"{experiment}", fontsize=18)
        ax.set_ylabel("Mean Values", fontsize=16)

    # Create a combined legend
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2)

    # Adjust layout and ensure legend is considered
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Save the figure with the legend
    plt.savefig("toy_summary_all.png", bbox_inches="tight")
    plt.show()


def summarize_cbo_toy_results_with_boxplots():
    selected_graphs6 = [
        rf"Graph6Unknown/.*_cbo_dr_EI_results_200_2",
        rf"Graph6/.*_cbo_results_200_2_graph_0",
        rf"Graph6/.*ceo_\d*_results_200_2_",
        rf"Graph6/.*_bo_results_new_200_2_",
        # rf"Graph6/.*_cbo_results_200_2_graph_4",
    ]
    selected_graphs5 = [
        rf"Graph5Unknown/.*_cbo_dr_EI_results_200_2",
        rf"Graph5/.*_cbo_results_200_2_graph_0",
        rf"Graph5/.*ceo_\d*_results_200_2_",
        rf"Graph5Wrong/.*_bo_results_new_200_2_",
        # rf"Graph5Wrong/.*_cbo_results_200_2_graph_0",
    ]
    selected_toy = [
        rf"ToyGraphUnknown/.*_cbo_dr_EI_results_200_2",
        rf"ToyGraph/.*_cbo_results_200_2_graph_0",
        rf"ToyGraph/.*ceo_\d*_results_200_2_",
        rf"ToyGraph/.*_bo_results_new_200_2_",
        # rf"ToyGraph/.*_cbo_results_200_2_graph_1",
    ]

    sns.set_theme(style="whitegrid")
    colors = ["blue", "orange"]
    # Gather data for box plots
    per_trial_data = {
        "ToyGraph": extract_per_trial_data(selected_toy),
        "Graph5": extract_per_trial_data(selected_graphs5),
        "Graph6": extract_per_trial_data(selected_graphs6),
    }

    best_y_data = {
        "ToyGraph": extract_best_y_data(selected_toy),
        "Graph5": extract_best_y_data(selected_graphs5),
        "Graph6": extract_best_y_data(selected_graphs6),
    }

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Experiment names for titles
    experiments = ["ToyGraph", "Graph5", "Graph6"]
    titles = ["Toy", "Epidemiology", "Healthcare"]

    for i, experiment in enumerate(experiments):
        # Combine data for both Per_trial_Y and Best_Y
        plot_data = prepare_combined_plot_data(
            per_trial_data[experiment], best_y_data[experiment]
        )

        # Create box plots for combined results
        sns.boxplot(
            ax=axs[i],
            x="Method",
            y="Value",
            hue="Type",
            data=plot_data,
            showfliers=False,
            palette=colors,
        )
        axs[i].set_title(titles[i])
        axs[i].set_ylabel("Values" if i == 0 else "")
        # axs[i].legend(loc="upper right")
        axs[i].get_legend().remove()  # Remove individual legends

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig("toy_summary_combined_boxplots.png", bbox_inches="tight")
    plt.show()


def extract_per_trial_data(regex_list):
    """
    Extracts all per trial Y data points from the results matching the regex list for use in box plots.
    """
    method_map = {0: "CBO Unknown", 1: "CBO", 2: "CEO", 3: "BO"}
    per_trial_data = {method: [] for method in method_map.values()}
    for i, regex in enumerate(regex_list):
        results = load_results_regex("", regex=regex)
        all_points = np.hstack([result["Per_trial_Y"][:30] for result in results])
        per_trial_data[method_map[i]].extend(all_points)

    return per_trial_data


def extract_best_y_data(regex_list):
    """
    Extracts the best Y data points from the results matching the regex list for use in box plots.
    """
    method_map = {0: "CBO Unknown", 1: "CBO", 2: "CEO", 3: "BO"}
    best_y_data = {method: [] for method in method_map.values()}
    for i, regex in enumerate(regex_list):
        results = load_results_regex("", regex=regex)
        best_results = np.vstack([result["Best_Y"][:30] for result in results]).min(
            axis=1
        )
        best_y_data[method_map[i]].extend(best_results)

    return best_y_data


def prepare_combined_plot_data(per_trial_data, best_y_data):
    """
    Prepares data for combined boxplot (Per_trial_Y and Best_Y) for a single experiment.
    """
    plot_data = {"Method": [], "Value": [], "Type": []}

    for method, values in per_trial_data.items():
        plot_data["Method"].extend([method] * len(values))
        plot_data["Value"].extend(values)
        plot_data["Type"].extend(["Average Y"] * len(values))

    for method, values in best_y_data.items():
        plot_data["Method"].extend([method] * len(values))
        plot_data["Value"].extend(values)
        plot_data["Type"].extend(["Best Y"] * len(values))

    return plot_data


def observational_dataset_posterior_per_iteration(nonlinear: bool = False):
    graphs = ["Erdos5", "Erdos6", "Erdos7", "Erdos8", "Erdos9", "Erdos10"]

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    linear_string = "Nonlinear" if nonlinear else "Linear"
    fig.suptitle(
        f"Posterior per Iteration for Different {linear_string} Graphs", fontsize=16
    )

    for i, graph in enumerate(graphs):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # Call the plot_posterior_nobs function with the current axis
        plot_posterior_nobs(graph, experiment="accuracy", nonlinear=nonlinear, ax=ax)

        ax.set_title(f"Graph: {graph}")

    # Add a single legend at the bottom of the figure
    handles, labels = axes[
        0, 0
    ].get_legend_handles_labels()  # Get handles and labels from one of the subplots
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(f"{linear_string}_per_iteration_accuracy", bbox_inches="tight")
    plt.show()
    # return fig
