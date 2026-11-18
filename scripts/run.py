"Run baseline BO benchmarks without template"

import argparse
import os

from tqdm import tqdm
import torch
import pandas as pd
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

from plateopt.gp import initialize_gp_model
from plateopt.acqf import optimise_qnei_and_get_suggestions
from plateopt.plate_designer import calculate_stock_solutions_from_df
from plateopt.utils import (unified_objective_function,
                                draw_sobol_samples,
                                get_index_from_lookup)


# benchmark datasets are loaded in main() for the selected --dataset only
PATH = './../benchmark_datasets'

tkwargs = {
    "dtype": torch.float64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}


def save_results(results, args):
    """
    Save results to a CSV file, appending if the file already exists.

    Args:
        results (dict): Dictionary containing the results of a single run.
        args (argparse.Namespace): Command-line arguments.
    """
    # Convert results dictionary to a DataFrame
    df = pd.DataFrame([results])

    # Create a descriptive filename
    filename = f"./../benchmark_results/results_{args.dataset}_baseline.csv"

    # Check if file already exists
    if os.path.exists(filename):
        # If it does, append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # If it doesn't, write the header
        df.to_csv(filename, index=False)


def main(args):
    "Run baseline BO with qNEI acquisition function."
    pl.seed_everything(args.seed)

    if args.dataset == 'buchwald':
        buchwald_benchmark = pd.read_csv(f'{PATH}/buchwald_virtual_benchmark.csv', index_col=0)
        # Separate features and targets
        X_df = buchwald_benchmark.iloc[:, 5:-2]
        y_df = buchwald_benchmark['AP yield']
        cond_df = buchwald_benchmark.iloc[:, :5]

    elif args.dataset == 'sulfonamide':
        sulfonamide_benchmark = pd.read_csv(f'{PATH}/sulfonamide_virtual_benchmark.csv', index_col=0)
        # Separate features and targets
        X_df = sulfonamide_benchmark.iloc[:, 5:-2]
        y_df = sulfonamide_benchmark['AP yield']

        # get rxn_id + string conditions
        cond_df = sulfonamide_benchmark.iloc[:, :5]

    else:
        raise ValueError("Invalid benchmark dataset.")

    # define X search space and Y total yields
    search_space = torch.tensor(X_df.to_numpy()).to(**tkwargs)
    yields = torch.tensor(y_df.to_numpy()).to(**tkwargs)
    max_yield = torch.max(yields).item()

    # from imported sobol function draw initial sobol samples
    train_x = draw_sobol_samples(n_samples=args.init_size, feature_matrix=search_space)
    train_y = unified_objective_function(train_x, search_space, yields)

    # calculate init stock solutions
    train_x_ids = get_index_from_lookup(train_x, search_space)
    train_x_df = cond_df.iloc[train_x_ids]
    train_x_df.drop('rxn_id', axis=1, inplace=True)
    init_stock_solns = calculate_stock_solutions_from_df(train_x_df)

    # init list to store
    best_yields = [(torch.max(train_y).item()/ max_yield) * 100]
    stock_solns_list = [init_stock_solns]

    for _ in tqdm(range(1, args.n_iterations+1)):
        # standardise train_y at every iteration
        scaler_y = StandardScaler()
        train_y_std = train_y.detach().clone()
        train_y_std = scaler_y.fit_transform(train_y_std.detach().cpu().numpy().reshape(-1, 1))
        train_y_std = torch.tensor(train_y_std).to(**tkwargs)

        model, _ = initialize_gp_model(train_x, train_y_std)

        with torch.no_grad():
            new_x = optimise_qnei_and_get_suggestions(
                model=model,
                train_x=train_x,
                search_matrix=search_space,
                batch_size=args.num_samples,
                seed=args.seed
            )

            new_x_ids = get_index_from_lookup(new_x, search_space)
            new_x_df = cond_df.iloc[new_x_ids]
            new_x_df.drop('rxn_id', axis=1, inplace=True)
            stock_solns = calculate_stock_solutions_from_df(new_x_df)
            stock_solns_list.append(stock_solns)

            new_y = unified_objective_function(new_x, search_space, yields)

            # get new training data
            train_y = torch.cat([train_y, new_y])
            train_x = torch.cat([train_x, new_x])

            # output max yield observed at each iteration
            best_yields.append((torch.max(train_y).item()/ max_yield) * 100)

    results = {
         'scoring_method': 'qnei',
         'dataset': args.dataset,
         'init_size': args.init_size,
         'batch_size': args.num_samples,
         'n_iterations': args.n_iterations,
         'seed': args.seed,
    }

    for i, best_yield in enumerate(best_yields):
        results[f"Max Normalized Yield (%) Iteration {i}"] = best_yield

    for i, stock_soln in enumerate(stock_solns_list):
        results[f"Stock solutions per Iteration {i}"] = stock_soln

    save_results(results, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline BO loop")
    parser.add_argument("--dataset", choices=["buchwald", "sulfonamide"], default="buchwald", help="Dataset to use for training")
    parser.add_argument("--init_size", default=12, type=int)
    parser.add_argument("--num_samples", default=48, type=int)
    parser.add_argument("--n_iterations", default=6, type=int)
    parser.add_argument("--seed", default=1, type=int)

    args = parser.parse_args()

    main(args)
