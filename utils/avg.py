import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True,
                        help="CSV file name (without extension) inside results/")
    parser.add_argument("--save", action="store_true",
                        help="Save the averaged results to a new CSV")

    args = parser.parse_args()

    PATH = "results"
    input_path = os.path.join(PATH, args.file + ".csv")

    # Load data
    df = pd.read_csv(input_path)

    # Drop non-metric columns
    df_metrics = df.drop(columns=["split", "n_times"], errors="ignore")

    # Compute column-wise mean
    avg = df_metrics.mean()

    # Convert to DataFrame for nicer formatting
    avg_df = avg.to_frame(name="mean").T

    print("\n=== AVERAGE RESULTS ===\n")
    print(avg_df.to_string(index=False))

    # Optionally save
    if args.save:
        output_path = os.path.join(PATH, args.file + "_avg.csv")
        avg_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

if __name__ == "__main__":
    main()