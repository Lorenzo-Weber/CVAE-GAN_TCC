import os
import csv

class Save():
    def save_result(
        self,
        split, n_times,

        # SVR
        mse_base, r2_base,
        mse_gan, r2_gan,

        # PLS
        mse_base_pls, r2_base_pls,
        mse_gan_pls, r2_gan_pls,

        # RF
        mse_base_rf, r2_base_rf,
        mse_gan_rf, r2_gan_rf,

        filename, run_id=''
    ):

        RESULTS_PATH = 'results'
        os.makedirs(RESULTS_PATH, exist_ok=True)

        file_path = os.path.join(RESULTS_PATH, f"{filename}_{run_id}.csv")

        file_exists = os.path.isfile(file_path)
        file_not_empty = file_exists and os.path.getsize(file_path) > 0

        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_not_empty:
                writer.writerow([
                    "split", "n_times",

                    # SVR
                    "mse_svr_base", "r2_svr_base",
                    "mse_svr_gan", "r2_svr_gan",

                    # PLS
                    "mse_pls_base", "r2_pls_base",
                    "mse_pls_gan", "r2_pls_gan",

                    # RF
                    "mse_rf_base", "r2_rf_base",
                    "mse_rf_gan", "r2_rf_gan",
                ])

            # Salvando os dados formatados com 4 casas decimais
            writer.writerow([
                split, n_times,

                # SVR
                f"{mse_base:.4f}", f"{r2_base:.4f}",
                f"{mse_gan:.4f}", f"{r2_gan:.4f}",

                # PLS
                f"{mse_base_pls:.4f}", f"{r2_base_pls:.4f}",
                f"{mse_gan_pls:.4f}", f"{r2_gan_pls:.4f}",

                # RF
                f"{mse_base_rf:.4f}", f"{r2_base_rf:.4f}",
                f"{mse_gan_rf:.4f}", f"{r2_gan_rf:.4f}",
            ])