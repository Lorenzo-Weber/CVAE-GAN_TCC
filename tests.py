import subprocess
from concurrent.futures import ProcessPoolExecutor
import os

MAX_GPU_PROCESSES = 2

def run_experiment(params):
    split, n_times = params

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # força usar mesma GPU

    print(f"Running split={split}, n_times={n_times}")

    subprocess.run(
        [
            "python3", "main.py",
            "--split", str(split),
            "--n_times", str(n_times)
        ],
        env=env
    )

if __name__ == "__main__":

    splits = [round(0.01 + i * 0.01, 2) for i in range(20)]
    n_times_list = list(range(2, 9))

    params = [(s, n) for s in splits for n in n_times_list]

    with ProcessPoolExecutor(max_workers=MAX_GPU_PROCESSES) as executor:
        executor.map(run_experiment, params)
