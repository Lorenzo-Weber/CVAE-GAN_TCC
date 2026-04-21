import argparse
import pandas as pd
import numpy as np
import os
import csv

import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from cvaegan.cvae_gan_trainer import GAN_trainer
from cvaegan.filter import Filter
from utils.plotter import Plotter
from utils.utils import SNV, MSC

def save_result(
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

        writer.writerow([
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
        ])

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--n_times", type=int, default=8)
    parser.add_argument("--split", type=float, default=0.14)

    parser.add_argument("--run_id", type=str)
    parser.add_argument("--save_results", action="store_true")

    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    N_TIMES = args.n_times
    SPLIT = args.split
    RUN_ID = args.run_id

    torch.manual_seed(42)
    np.random.seed(42)
    os.makedirs('figs/', exist_ok=True)

    PATH = os.path.join('data', 'soilNIR')
    FILE_NAME = 'DataResearch'
    DATA_TYPE = '.xlsx'
    SHEET_NAME = 'Raw spectral'
    DS = os.path.join(PATH, FILE_NAME + DATA_TYPE)

    if DATA_TYPE == '.csv':
        df = pd.read_csv(DS)
    else:
        df = pd.read_excel(DS, sheet_name=SHEET_NAME)

    x = df.iloc[:, 1:-2].to_numpy(dtype=np.float32)
    y = df.iloc[:, -2:].to_numpy(dtype=np.float32)

    print(f'Features shape: {x.shape}')
    print(f'Labels shape: {y.shape}')

    NUM_FEATURES = x.shape[1]
    NUM_CONDITIONS = y.shape[1]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    snv = SNV()
    msc = MSC()

    x_train_scaled = scaler_x.fit_transform(x_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    x_train_snv = snv.fit_transform(x_train_scaled)

    x_tensor = torch.tensor(x_train_snv, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

    dataset = TensorDataset(x_tensor, y_tensor)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    gan_trainer = GAN_trainer(
        num_conditions=NUM_CONDITIONS,
        data_length=NUM_FEATURES,
        batch_size=BATCH_SIZE,
    )

    gan_trainer.train(train_loader, val_loader, epochs=EPOCHS)

    x_gan, y_gan = gan_trainer.generate(train_loader, n_times=N_TIMES)

    x_gan = x_gan.numpy()
    y_gan = y_gan.numpy()

    x_gan = snv.inverse_transform(x_gan)
    x_gan = scaler_x.inverse_transform(x_gan)
    y_gan = scaler_y.inverse_transform(y_gan)

    msc.fit(x_train)
    x_gan = msc.transform(x_gan)

    filter = Filter(split=SPLIT)
    x_filtered, y_filtered = filter.filter(x_gan, y_gan, x_train)

    plotter = Plotter()
    plotter.compare_real_vs_generated(x_train, x_filtered, n_samples=4, filename=FILE_NAME)

    if isinstance(x_filtered, torch.Tensor):
        x_filtered = x_filtered.detach().cpu().numpy()

    if isinstance(y_filtered, torch.Tensor):
        y_filtered = y_filtered.detach().cpu().numpy()

    if x_filtered.ndim == 3:
        x_filtered = np.squeeze(x_filtered, axis=1)

    x_train_aug = np.concatenate([x_train, x_filtered], axis=0)
    y_train_aug = np.concatenate([y_train, y_filtered], axis=0)

    # ===================== SVR =====================
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", MultiOutputRegressor(SVR(kernel='poly')))
    ])

    model.fit(x_train, y_train)
    y_pred_base = model.predict(x_test)

    mse_base = mean_squared_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)

    print('--- SVR BASE ---')
    print("MSE:", mse_base)
    print("R2:", r2_base)

    model.fit(x_train_aug, y_train_aug)
    y_pred_gan = model.predict(x_test)

    mse_gan = mean_squared_error(y_test, y_pred_gan)
    r2_gan = r2_score(y_test, y_pred_gan)

    print('--- SVR + GAN ---')
    print("MSE:", mse_gan)
    print("R2:", r2_gan)

    # ===================== PLS =====================
    n_components = min(10, x_train.shape[0] - 1)

    pls_model = Pipeline([
        ("scaler", StandardScaler()),
        ("pls", PLSRegression(n_components=n_components))
    ])

    pls_model.fit(x_train, y_train)
    y_pred_base_pls = pls_model.predict(x_test)

    mse_base_pls = mean_squared_error(y_test, y_pred_base_pls)
    r2_base_pls = r2_score(y_test, y_pred_base_pls)

    print('--- PLS BASE ---')
    print("MSE:", mse_base_pls)
    print("R2:", r2_base_pls)

    pls_model.fit(x_train_aug, y_train_aug)
    y_pred_gan_pls = pls_model.predict(x_test)

    mse_gan_pls = mean_squared_error(y_test, y_pred_gan_pls)
    r2_gan_pls = r2_score(y_test, y_pred_gan_pls)

    print('--- PLS + GAN ---')
    print("MSE:", mse_gan_pls)
    print("R2:", r2_gan_pls)

    # ===================== RANDOM FOREST =====================
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    rf.fit(x_train, y_train)
    y_pred_base_rf = rf.predict(x_test)

    mse_base_rf = mean_squared_error(y_test, y_pred_base_rf)
    r2_base_rf = r2_score(y_test, y_pred_base_rf)

    print('--- RF BASE ---')
    print("MSE:", mse_base_rf)
    print("R2:", r2_base_rf)

    rf.fit(x_train_aug, y_train_aug)
    y_pred_gan_rf = rf.predict(x_test)

    mse_gan_rf = mean_squared_error(y_test, y_pred_gan_rf)
    r2_gan_rf = r2_score(y_test, y_pred_gan_rf)

    print('--- RF + GAN ---')
    print("MSE:", mse_gan_rf)
    print("R2:", r2_gan_rf)

    if args.save_results:
        save_result(
            SPLIT, N_TIMES,

            # SVR
            mse_base, r2_base,
            mse_gan, r2_gan,

            # PLS
            mse_base_pls, r2_base_pls,
            mse_gan_pls, r2_gan_pls,

            # RF
            mse_base_rf, r2_base_rf,
            mse_gan_rf, r2_gan_rf,

            filename=FILE_NAME,
            run_id=RUN_ID
)

if __name__ == "__main__":
    main()