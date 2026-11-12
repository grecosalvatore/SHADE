# Drift severity and score correlation (m_w = 1000, mean over 5 runs)

rho: Spearman correlation between the drift percentage of a window and its score, over the windows with 0-100% and 0-20% drift. Onset: smallest drift percentage at which the mean score of the windows exceeds the threshold. Better value of each DriftLens / SHADE pair in bold.

| UC | DriftLens rho 0-100% | DriftLens rho 0-20% | DriftLens onset | SHADE rho 0-100% | SHADE rho 0-20% | SHADE onset |
|---|---|---|---|---|---|---|
| UC01 | **0.999** | **0.980** | **5%** | **0.999** | 0.978 | **5%** |
| UC02 | **0.998** | **0.961** | **5%** | **0.998** | 0.930 | 10% |
| UC03 | **0.999** | **0.967** | **5%** | 0.998 | 0.944 | 10% |
| UC04 | **0.999** | 0.979 | **5%** | **0.999** | **0.980** | **5%** |
| UC05 | **0.999** | **0.979** | 10% | **0.999** | **0.979** | **5%** |
| UC06 | **0.999** | **0.975** | **5%** | **0.999** | 0.959 | 10% |
| UC07 | **0.999** | **0.980** | **5%** | **0.999** | **0.980** | **5%** |
| UC08 | **0.999** | **0.980** | **5%** | 0.998 | 0.943 | 10% |
| UC09 | **0.972** | **0.658** | **25%** | 0.403 | -0.012 | 100% |
| UC10 | 0.973 | 0.652 | 15% | **0.987** | **0.885** | **5%** |
| **Mean** | **0.994** | **0.911** | -- | 0.938 | 0.857 | -- |
| **Median** | **0.999** | **0.977** | **5%** | 0.998 | 0.952 | 7.5% |
