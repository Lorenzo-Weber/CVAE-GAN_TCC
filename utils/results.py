import pandas as pd
import argparse
import os

def print_metrics(results, model_prefix, display_name):
    print(f'{"="*15} {display_name} {"="*15}')

    for variant in ['base', 'gan', 'shift']:
        mse_col = f'mse_{model_prefix}_{variant}'
        r2_col = f'r2_{model_prefix}_{variant}'

        best_mse_idx = results[mse_col].idxmin()
        best_r2_idx = results[r2_col].idxmax()

        best_mse_row = results.loc[best_mse_idx]
        best_r2_row = results.loc[best_r2_idx]

        print(f'--- {variant.upper()} ---')
        print(f'Best MSE Score: {best_mse_row[mse_col]:.4f}')
        print(f'split: {best_mse_row["split"]}, \t\t n_times: {int(best_mse_row["n_times"])}')
        print(f'R2 score with this config: {best_mse_row[r2_col]:.4f}')
        print()
        
        print(f'Best R2 Score: {best_r2_row[r2_col]:.4f}')
        print(f'split: {best_r2_row["split"]}, \t\t n_times: {int(best_r2_row["n_times"])}')
        print(f'MSE Score with this config: {best_r2_row[mse_col]:.4f}')
        print()

    print("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help='Choose a file to show the results from (full name without file extension)')

    args = parser.parse_args()

    PATH = 'results'
    FILE = args.file
    
    RESULTS_PATH = os.path.join(PATH, FILE + '.csv')

    results = pd.read_csv(RESULTS_PATH)

    # Chamada da função para cada modelo
    print_metrics(results, 'svr', 'SVR')
    print_metrics(results, 'pls', 'PLS')
    print_metrics(results, 'rf', 'RANDOM FOREST')

if __name__ == '__main__':
    main()