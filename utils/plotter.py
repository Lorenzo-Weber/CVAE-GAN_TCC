import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

class Plotter:
    def __init__(self, save_dir="figs", random_state=42):
        self.save_dir = save_dir
        self.rng = np.random.default_rng(random_state)

        os.makedirs(save_dir, exist_ok=True)

    def _to_numpy(self, x):
        if isinstance(x, pd.DataFrame):
            return x.to_numpy()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def compare_real_vs_generated(
        self,
        x_real,
        x_fake,
        filename = '',
        n_samples=10,
        same_scale=True
    ):
        x_real = self._to_numpy(x_real)
        x_fake = self._to_numpy(x_fake)

        n_samples = min(n_samples, len(x_real), len(x_fake))

        idx_real = self.rng.choice(len(x_real), n_samples, replace=False)
        idx_fake = self.rng.choice(len(x_fake), n_samples, replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Reais
        for i in idx_real:
            axes[0].plot(x_real[i], alpha=0.7)

        axes[0].set_title("Espectros Reais")
        axes[0].set_xlabel("Comprimento do espectro")
        axes[0].set_ylabel("Intensidade")

        # Fakes
        for i in idx_fake:
            axes[1].plot(x_fake[i], alpha=0.7)

        axes[1].set_title("Espectros Gerados (Filtrados)")
        axes[1].set_xlabel("Comprimento do espectro")

        # mesma escala para comparação justa
        if same_scale:
            ymin = min(x_real.min(), x_fake.min())
            ymax = max(x_real.max(), x_fake.max())
            axes[0].set_ylim(ymin, ymax)
            axes[1].set_ylim(ymin, ymax)

        plt.tight_layout()

        path = os.path.join(self.save_dir, filename + '.png')
        plt.savefig(path)
        plt.close()

        print(f"Figura salva em: {path}")