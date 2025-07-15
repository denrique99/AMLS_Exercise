import numpy as np

def pla(signal, seg_len=30):          # 30 samples ≈ 0.1 s @300 Hz
    n = len(signal)
    points = np.arange(0, n, seg_len)
    if points[-1] != n-1: points = np.append(points, n-1)

    compressed = []
    for i in range(len(points)-1):
        s, e = points[i], points[i+1]
        x = np.linspace(0, 1, e-s+1)
        y = np.interp(x, [0,1], [signal[s], signal[e]])
        compressed.append(y.astype(np.int16))  # keep dtype for simplicity
    return np.concatenate(compressed)
