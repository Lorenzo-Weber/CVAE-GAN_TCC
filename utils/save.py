import os
import csv
import json


class Save:
    def save_result(
        self,
        split, n_times,

        # SVR
        mse_base, r2_base, mae_base,
        mse_gan, r2_gan, mae_gan,
        mse_shift, r2_shift, mae_shift,

        # PLS
        mse_base_pls, r2_base_pls, mae_base_pls,
        mse_gan_pls, r2_gan_pls, mae_gan_pls,
        mse_shift_pls, r2_shift_pls, mae_shift_pls,

        # RF
        mse_base_rf, r2_base_rf, mae_base_rf,
        mse_gan_rf, r2_gan_rf, mae_gan_rf,
        mse_shift_rf, r2_shift_rf, mae_shift_rf,

        filename,
        architecture,
        run_id=''
    ):
        RESULTS_PATH = 'results'
        os.makedirs(RESULTS_PATH, exist_ok=True)

        file_path = os.path.join(RESULTS_PATH, f"{filename}_{run_id}.csv")

        file_exists = os.path.isfile(file_path)
        file_not_empty = file_exists and os.path.getsize(file_path) > 0

        gen_layers, disc_layers = architecture

        gen_layers_json = json.dumps(gen_layers)
        disc_layers_json = json.dumps(disc_layers)

        def f(x):
            return f"{x:.4f}"

        with open(file_path, "a", newline="") as f_csv:
            writer = csv.writer(f_csv)

            if not file_not_empty:
                writer.writerow([
                    "split", "n_times", "generator", "discriminator",

                    # SVR
                    "mse_svr_base", "r2_svr_base", "mae_svr_base",
                    "mse_svr_gan", "r2_svr_gan", "mae_svr_gan",
                    "mse_svr_shift", "r2_svr_shift", "mae_svr_shift",

                    # PLS
                    "mse_pls_base", "r2_pls_base", "mae_pls_base",
                    "mse_pls_gan", "r2_pls_gan", "mae_pls_gan",
                    "mse_pls_shift", "r2_pls_shift", "mae_pls_shift",

                    # RF
                    "mse_rf_base", "r2_rf_base", "mae_rf_base",
                    "mse_rf_gan", "r2_rf_gan", "mae_rf_gan",
                    "mse_rf_shift", "r2_rf_shift", "mae_rf_shift",
                ])

            writer.writerow([
                split,
                n_times,
                gen_layers_json,
                disc_layers_json,

                # SVR
                f(mse_base), f(r2_base), f(mae_base),
                f(mse_gan), f(r2_gan), f(mae_gan),
                f(mse_shift), f(r2_shift), f(mae_shift),

                # PLS
                f(mse_base_pls), f(r2_base_pls), f(mae_base_pls),
                f(mse_gan_pls), f(r2_gan_pls), f(mae_gan_pls),
                f(mse_shift_pls), f(r2_shift_pls), f(mae_shift_pls),

                # RF
                f(mse_base_rf), f(r2_base_rf), f(mae_base_rf),
                f(mse_gan_rf), f(r2_gan_rf), f(mae_gan_rf),
                f(mse_shift_rf), f(r2_shift_rf), f(mae_shift_rf),
            ])