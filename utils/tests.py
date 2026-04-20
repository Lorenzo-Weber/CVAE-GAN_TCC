import subprocess
from concurrent.futures import ProcessPoolExecutor
import os
import uuid

MAX_GPU_PROCESSES = 2

def run_experiment(params):
    split, n_times, run_id = params

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"Running split={split}, n_times={n_times}, run_id={run_id}")

    subprocess.run(
        [
            "python3", "main.py",
            "--split", str(split),
            "--n_times", str(n_times),
            "--save_results",
            "--run_id", run_id
        ],
        env=env,
        check=True
    )

if __name__ == "__main__":

    run_id = uuid.uuid4().hex[:8]

    splits = [round(0.01 + i * 0.01, 2) for i in range(20)]
    n_times_list = list(range(2, 9))

    params = [(s, n, run_id) for s in splits for n in n_times_list]

    with ProcessPoolExecutor(max_workers=MAX_GPU_PROCESSES) as executor:
        executor.map(run_experiment, params)