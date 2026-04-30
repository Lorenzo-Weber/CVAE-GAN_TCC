import pandas as pd
import argparse
import os
import json


def print_metrics(results, model_prefix, display_name):
    print(f'{"="*15} {display_name} {"="*15}')

    for variant in ['base', 'gan', 'shift']:
        mse_col = f'mse_{model_prefix}_{variant}'
        r2_col = f'r2_{model_prefix}_{variant}'
        mae_col = f'mae_{model_prefix}_{variant}'

        # --- Melhor MSE ---
        best_mse_idx = results[mse_col].idxmin()
        best_mse_row = results.loc[best_mse_idx]

        # --- Melhor R2 ---
        best_r2_idx = results[r2_col].idxmax()
        best_r2_row = results.loc[best_r2_idx]

        # --- Parse arquitetura ---
        gen_mse = json.loads(best_mse_row["generator"])
        disc_mse = json.loads(best_mse_row["discriminator"])

        gen_r2 = json.loads(best_r2_row["generator"])
        disc_r2 = json.loads(best_r2_row["discriminator"])

        print(f'--- {variant.upper()} ---')

        print(f'Best MSE Score: {best_mse_row[mse_col]:.4f}')
        print(f'MAE: {best_mse_row[mae_col]:.4f}')
        print(f'R2 (same config): {best_mse_row[r2_col]:.4f}')
        print(f'split: {best_mse_row["split"]}, n_times: {int(best_mse_row["n_times"])}')
        print(f'Generator: {gen_mse}')
        print(f'Discriminator: {disc_mse}')
        print()

        print(f'Best R2 Score: {best_r2_row[r2_col]:.4f}')
        print(f'MAE: {best_r2_row[mae_col]:.4f}')
        print(f'MSE (same config): {best_r2_row[mse_col]:.4f}')
        print(f'split: {best_r2_row["split"]}, n_times: {int(best_r2_row["n_times"])}')
        print(f'Generator: {gen_r2}')
        print(f'Discriminator: {disc_r2}')
        print()

    print("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help='Nome do arquivo (sem .csv)'
    )

    args = parser.parse_args()

    PATH = 'results'
    FILE = args.file
    RESULTS_PATH = os.path.join(PATH, FILE + '.csv')

    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {RESULTS_PATH}")

    results = pd.read_csv(RESULTS_PATH)

    # --- Garantir tipos numéricos ---
    numeric_cols = [col for col in results.columns if col.startswith(("mse_", "r2_", "mae_"))]
    results[numeric_cols] = results[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # --- Execução ---
    print_metrics(results, 'svr', 'SVR')
    print_metrics(results, 'pls', 'PLS')
    print_metrics(results, 'rf', 'RANDOM FOREST')


if __name__ == '__main__':
    main()