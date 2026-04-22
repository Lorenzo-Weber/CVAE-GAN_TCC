import subprocess

for _ in range(50):
    subprocess.run(
        [
            "python3", "main.py",
            "--split", str(0.05),
            "--n_times", str(3),
            "--save_results",
            "--run_id", 'pls'
        ],
        check=True
    )