import numpy as np

def augment_with_shift(X, y, shifts=(-2, -1, 1, 2), fill_value=0.0):
    """
    Aplica data augmentation por deslocamento espectral.

    Parâmetros:
        X: array (n_samples, n_features)
        y: array (n_samples, ...) ou None
        shifts: iterable com valores de shift (int ou float)
        fill_value: valor para bordas após deslocamento

    Retorna:
        X_aug, y_aug
    """

    X = np.asarray(X)

    X_aug = [X]
    y_aug = [y] if y is not None else None

    n_features = X.shape[1]
    base_idx = np.arange(n_features)

    for s in shifts:
        shifted = np.empty_like(X)

        shifted_idx = base_idx - s

        for i in range(X.shape[0]):
            shifted[i] = np.interp(
                base_idx,
                shifted_idx,
                X[i],
                left=fill_value,
                right=fill_value
            )

        X_aug.append(shifted)

        if y is not None:
            y_aug.append(y)

    X_aug = np.vstack(X_aug)

    if y is not None:
        y_aug = np.vstack(y_aug) if y.ndim > 1 else np.concatenate(y_aug)
        return X_aug, y_aug

    return X_aug