# SHADE variants: K random directions, B bins per direction

Mean over all use cases and window sizes (5 runs each). Seed std: standard deviation of AUC@5% over the runs of each use case and window size, averaged. Stored: reference state, mean over use cases. ms/window: median time per window measured during the detection runs. Best SHADE value per column in bold; DriftLens as reference.

| Variant | AUC@5% | AUC@10% | Acc@0% | H_DD | Seed std | ms/window | Stored (kB) |
|---|---|---|---|---|---|---|---|
| K = 64, B = 2 (default) | 0.841 | 0.923 | 0.918 | 0.683 | 0.027 | **2.48** | **7.1** |
| K = 64, B = 8 | 0.845 | 0.929 | **0.935** | 0.691 | 0.024 | 2.98 | 29.5 |
| K = 128, B = 2 | 0.847 | 0.926 | 0.887 | **0.722** | 0.017 | 4.27 | 14.2 |
| K = 128, B = 8 | **0.851** | **0.933** | 0.896 | **0.722** | **0.016** | 5.39 | 59.1 |
| DriftLens | 0.868 | 0.944 | 0.953 | 0.730 | 0.005 | 16.40 | 2048.4 |
