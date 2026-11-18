"Run benchmarks of templated BO."

import argparse
import os

import torch
import pandas as pd
import pytorch_lightning as pl

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from plateopt.gp import initialize_gp_model
from plateopt.utils import unified_objective_function, draw_sobol_samples, get_index_from_lookup
from plateopt.scoring import score_reactions
from plateopt.mapping_utils import ReactionScoreAnalyzer
from plateopt.plate_designer import HTEPlateDesigner, calculate_stock_solutions_from_df


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
    filename = f"./../benchmark_results/results_{args.dataset}_template.csv"

    # Check if file already exists
    if os.path.exists(filename):
        # If it does, append without writing the header
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        # If it doesn't, write the header
        df.to_csv(filename, index=False)


def main(args):
    "Run templated Bayesian optimisation "
    pl.seed_everything(args.seed)

    if args.dataset == 'buchwald':
        buchwald_benchmark = pd.read_csv(f'{PATH}/buchwald_virtual_benchmark.csv', index_col=0)
        # Separate features and targets
        X_df = buchwald_benchmark.iloc[:, 5:-2]
        y_df = buchwald_benchmark['AP yield']

        # get rxn_id + string conditions
        cond_df = buchwald_benchmark.iloc[:, :5]
        rxn_ids = buchwald_benchmark['rxn_id']

    elif args.dataset == 'sulfonamide':
        sulfonamide_benchmark = pd.read_csv(f'{PATH}/sulfonamide_virtual_benchmark.csv', index_col=0)
        # Separate features and targets
        X_df = sulfonamide_benchmark.iloc[:, 5:-2]
        y_df = sulfonamide_benchmark['AP yield']

        # get rxn_id + string conditions
        cond_df = sulfonamide_benchmark.iloc[:, :5]
        rxn_ids = sulfonamide_benchmark['rxn_id']

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
    train_x_df.drop('rxn_id', axis=1, inplace=True) # exclude rxn_id from stock_soln calculation
    init_stock_solns = calculate_stock_solutions_from_df(train_x_df)

    # init list to store
    best_yields = [(torch.max(train_y).item()/ max_yield) * 100]
    stock_solns_list = [init_stock_solns]

    for _ in tqdm(range(1, args.n_iterations+1)):
        # standardise train_y at every iteration
        scaler_y = StandardScaler()
        train_y_std = train_y.detach().clone()
        train_y_std = scaler_y.fit_transform(train_y_std.detach().cpu().numpy().reshape(-1,1))
        train_y_std = torch.tensor(train_y_std).to(**tkwargs)

        model, _ = initialize_gp_model(train_x, train_y_std)

        # score all reaction conditions in the search space
        scores = score_reactions(gp_model=model,
                                 train_x=train_x,
                                 x_space=search_space,
                                 seed=args.seed,
                                 method=args.scoring_method)
        # update at every iteration
        scores_df = pd.DataFrame({
            'rxn_id': rxn_ids,
            'score': scores
            })

        # create analyzer object with dataframe conditions and scores
        analyzer = ReactionScoreAnalyzer(conditions_df=cond_df, scores_df=scores_df)

        # create designer object to find optimal HTE plate designs
        designer = HTEPlateDesigner(score_analyzer=analyzer, plate_sizes=[args.num_samples])

        top_designs = designer.find_best_plate_designs(
            metric=args.metric, n_designs=100)

        # Filter designs to only those with stock solution count obeying threshold
        filtered_designs = [
            design for design in top_designs if design['stock_solns'] <= args.threshold
            ]

        # Take the best design from filtered results
        best_plate_design = filtered_designs[0]

        best_decomp = best_plate_design['decomposition']
        stock_solns = best_plate_design['stock_solns']
        stock_solns_list.append(stock_solns)
        print(stock_solns)

        best_rxn_ids = designer.get_plate_design_rxn_ids(best_decomp, metric=args.metric)
        best_rxn_ids = torch.tensor(best_rxn_ids, dtype=torch.long)

        # rxn_id equals the row index in search_space
        new_x = search_space[best_rxn_ids]
        new_y = yields[best_rxn_ids]

        # get new training data
        train_y = torch.cat([train_y, new_y])
        train_x = torch.cat([train_x, new_x])

        print(train_x.size())

        # output max yield observed at each iteration
        best_yields.append((torch.max(train_y).item()/ max_yield) * 100)

    results = {
         'scoring_method': args.scoring_method,
         'agg_metric': args.metric,
         'dataset': args.dataset,
         'stock_soln_threshold': args.threshold,
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
    parser = argparse.ArgumentParser(description="Template BO loop")
    parser.add_argument("--dataset", choices=["buchwald", "sulfonamide"], default="buchwald", help="Dataset to use for training")
    parser.add_argument("--threshold", default=30, type=int)
    parser.add_argument("--init_size", default=12, type=int)
    parser.add_argument("--num_samples", default=48, type=int)
    parser.add_argument("--n_iterations", default=6, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--scoring_method", choices=["qnei", "ucb", "qpo"], default="qnei")
    parser.add_argument("--metric", default="mean")

    args = parser.parse_args()

    main(args)
