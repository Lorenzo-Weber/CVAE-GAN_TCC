import pandas as pd
import argparse
import os

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help='Choose a file to show the results from (full name without file extension)')

    args = parser.parse_args()

    PATH = 'results'
    FILE = args.file
    
    RESULTS_PATH = os.path.join(PATH, FILE + '.csv')

    results = pd.read_csv(RESULTS_PATH)

    # Gets the index of the lowest value and the oposite for the r2 score
    best_mse_idx = results['mse_gan'].idxmin()
    best_r2_idx = results['r2_gan'].idxmax()

    # Locates the actual values from that index
    best_mse_row = results.loc[best_mse_idx]
    best_r2_row = results.loc[best_r2_idx]

    # Prints it out
    print(f'Best MSE Score: {best_mse_row["mse_gan"]:.4f}')
    print(f'split: {best_mse_row["split"]}, \t\t n_times: {int(best_mse_row["n_times"])}')
    print(f'R2 score with this config: {best_mse_row["r2_gan"]:.4f}')

    print()

    print(f'Best R2 Score: {best_r2_row["r2_gan"]:.4f}')
    print(f'split: {best_r2_row["split"]}, \t\t n_times: {int(best_r2_row["n_times"])}')
    print(f'MSE Score with this config: {best_r2_row["mse_gan"]:.4f}')

if __name__ == '__main__':
    main()