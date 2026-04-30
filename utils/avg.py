import pandas as pd
import argparse
import os
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="CSV file name (without extension) inside results/"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the averaged results to a new CSV"
    )

    args = parser.parse_args()

    PATH = "results"
    input_path = os.path.join(PATH, args.file + ".csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    # --- Load data ---
    df = pd.read_csv(input_path)

    # --- Garantir que métricas são numéricas ---
    metric_cols = [col for col in df.columns if col.startswith(("mse_", "r2_", "mae_"))]
    df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")

    # --- Validar presença das colunas de arquitetura ---
    if "generator" not in df.columns or "discriminator" not in df.columns:
        raise ValueError("CSV não contém colunas 'generator' e 'discriminator'")

    # --- Normalizar JSON (opcional, mas robusto) ---
    df["generator"] = df["generator"].apply(lambda x: json.dumps(json.loads(x)))
    df["discriminator"] = df["discriminator"].apply(lambda x: json.dumps(json.loads(x)))

    # --- Agrupar por configuração ---
    group_cols = ["generator", "discriminator", "split", "n_times"]

    df_avg = (
        df.groupby(group_cols, as_index=False)[metric_cols]
        .mean()
        .sort_values(by="mse_svr_base")  # critério default
    )

    print("\n=== AVERAGED RESULTS (per configuration) ===\n")
    print(df_avg.to_string(index=False))

    # --- Média global (opcional, mas útil) ---
    global_avg = df[metric_cols].mean().to_frame(name="mean").T

    print("\n=== GLOBAL AVERAGE (sanity check) ===\n")
    print(global_avg.to_string(index=False))

    # --- Save ---
    if args.save:
        output_path = os.path.join(PATH, args.file + "_avg.csv")
        df_avg.to_csv(output_path, index=False)
        print(f"\nSaved grouped results to: {output_path}")

        global_path = os.path.join(PATH, args.file + "_global_avg.csv")
        global_avg.to_csv(global_path, index=False)
        print(f"Saved global average to: {global_path}")


if __name__ == "__main__":
    main()