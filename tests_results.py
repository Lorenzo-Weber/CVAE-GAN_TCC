import pandas as pd

results = pd.read_csv('results.csv')

# Gets the index of the lowest value and the oposite for the r2 score
best_mse_idx = results['mse_gan'].idxmin()
best_r2_idx = results['r2_gan'].idxmax()

# Locates the actual values from that index
best_mse_row = results.loc[best_mse_idx]
best_r2_row = results.loc[best_r2_idx]

# Prints it out
print(f'Best MSE Score: {best_mse_row["mse_gan"]}')
print(f'split: {best_mse_row["split"]}, n_times: {best_mse_row["n_times"]}')

print()

print(f'Best R2 Score: {best_r2_row["r2_gan"]}')
print(f'split: {best_r2_row["split"]}, n_times: {best_r2_row["n_times"]}')