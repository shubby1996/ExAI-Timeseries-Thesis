Creating a plan for addressing the question - 
Does putting uncertainty in models like NHIts and TCN which are inherently deterministic reduce there accuracy or increase it?

We will run out commands in data/.env 
Activate the .env

1. Sanity / baseline: ensure preprocessing & results folder are up-to-date
``` data/.env/Scripts/python.exe run_compare_nhits_timesnet.py --full --help ```

2. Run deterministic (MAE) models evaluation for both architectures
``` data/.env/Scripts/python.exe NHiTS/train_nhits_deterministic.py   # if you want to train ```
```data/.env/Scripts/python.exe NHiTS/predict_nhits_nordbyen.py```
```data/.env/Scripts/python.exe NHiTS/evaluate_nhits_nordbyen.py ```

``` data/.env/Scripts/python.exe timesnet/train_timesnet_mae.py      # train if needed ```
```data/.env/Scripts/python.exe timesnet/predict_timesnet_nordbyen.py --model_name timesnet_deterministic_mae ```
```data/.env/Scripts/python.exe timesnet/evaluate_timesnet_nordbyen.py --model_name timesnet_deterministic_mae```

Inspect after each evaluate:
Files: 
```data/results/nhits_evaluation_metrics.csv, data/results/nhits_evaluation_predictions.csv```
Files: 
```data/results/timesnet_evaluation_metrics.csv, data/results/timesnet_evaluation_predictions.csv```

Key numbers: MAE, RMSE, MAPE, R² — record these.

3. Run MSE experiments

```data/.env/Scripts/python.exe NHiTS/train_nhits_nordbyen.py  ```
```data/.env/Scripts/python.exe NHiTS/predict_nhits_nordbyen.py```
```data/.env/Scripts/python.exe NHiTS/evaluate_nhits_nordbyen.py```

```data/.env/Scripts/python.exe timesnet/train_timesnet_mse.py```
```data/.env/Scripts/python.exe timesnet/predict_timesnet_nordbyen.py --model_name timesnet_mse```
```data/.env/Scripts/python.exe timesnet/evaluate_timesnet_nordbyen.py --model_name timesnet_mse```

Inspect:
data/results/timesnet_evaluation_metrics.csv (overwrites or create separate file — if overwrite risk, copy results with timestamp)
Compare MSE-trained TimesNet MAE/RMSE vs MAE-trained TimesNet.

4. Run probabilistic/quantile evaluation for both architectures

```data/.env/Scripts/python.exe NHiTS/train_nhits_probabilistic.py   # or use saved nhits_probabilistic_q model```
```data/.env/Scripts/python.exe NHiTS/predict_nhits_nordbyen.py      # ensure it picks the probabilistic model or pass args```
```data/.env/Scripts/python.exe NHiTS/evaluate_nhits_nordbyen.py```

```data/.env/Scripts/python.exe timesnet/train_timesnet_prob.py```
```data/.env/Scripts/python.exe timesnet/predict_timesnet_nordbyen.py --model_name timesnet_probabilistic_q```
```data/.env/Scripts/python.exe timesnet/evaluate_timesnet_nordbyen.py --model_name timesnet_probabilistic_q```

Inspect:
Predictions: files (for probabilistic models) will include quantiles or sample columns. Check data/results/*_evaluation_predictions.csv.
Metrics: those evaluate scripts compute MAE, RMSE and also report avg quantile loss / PICP / MIW if implemented — record:
MAE (use median as point forecast)
Avg quantile loss or CRPS (approx via average pinball loss)
PICP at chosen nominal (e.g., 80%) and MIW (interval width)
 
5. Aggregate comparisons and statistical testing
Steps (can be Python snippet or Jupyter):
Load the metrics CSVs for each model/variant.
Build a table: rows = model variants (NHiTS-deterministic, NHiTS-prob, TimesNet-MSE, TimesNet-MAE, TimesNet-prob), columns = MAE, RMSE, MAPE, AvgQuantileLoss/CRPS, PICP@80, MIW@80, training time.
Statistical test for point-forecast differences (e.g., bootstrap test or Diebold–Mariano on MAE series across test windows).
Quick command (run a small Python script I can prepare) — I can prepare it if you want.

6. Interpret and conclude
How to read results:
If probabilistic median MAE < deterministic MAE (statistically significant) → probabilistic training improved point accuracy.
If probabilistic median MAE ≈ deterministic MAE but CRPS / quantile loss and coverage are better → probabilistic yields better uncertainty while preserving accuracy.
If median MAE worse but probabilistic scores much better and coverage improved → trade-off exists: better uncertainty at cost of point accuracy.
Also examine training cost: if a probabilistic model needs much more compute, factor that in.

==================================================================


1. Train NHiTS with MAE Loss
```python NHiTS/train_nhits_mse.py```

What to Expect:
This command trains the NHiTS model using the Mean Absolute Error (MAE) loss function.
The training logs will show the progress of the model, including epoch-wise loss values and validation metrics.

What to Check:
Verify that the model checkpoints (e.g., nhits_nordbyen.pt) are saved in the models directory.
Check the training logs for convergence (loss decreasing over epochs).

2. Train NHiTS with MSE Loss
```python NHiTS/train_nhits_nordbyen.py --loss mse```

What to Expect:
This command trains the NHiTS model using the Mean Squared Error (MSE) loss function.
Similar to the MAE training, logs will display epoch-wise loss and validation metrics.

What to Check:
Ensure that the MSE-trained model checkpoints (e.g., nhits_nordbyen_mse.pt) are saved in the models directory.
Confirm that the training loss and validation metrics are logged correctly.

3. Train NHiTS with Probabilistic Loss
```python NHiTS/train_nhits_nordbyen.py --loss probabilistic```

What to Expect:
This command trains the NHiTS model with a probabilistic loss function, enabling the model to predict uncertainty.
Logs will include probabilistic metrics like quantile loss.

What to Check:
Verify that the probabilistic model checkpoints (e.g., nhits_prob.pt) are saved in the models directory.
Check the logs for probabilistic metrics and ensure they improve over epochs.

4. Train TimesNet with MAE Loss
```python timesnet/train_timesnet_mae.py```
What to Expect:

This command trains the TimesNet model using the MAE loss function.
Logs will display epoch-wise loss and validation metrics.
What to Check:

Ensure that the MAE-trained TimesNet model checkpoints (e.g., timesnet_nordbyen.pt) are saved in the models directory.
Confirm that the training logs show decreasing loss values.

5. Train TimesNet with MSE Loss
```python timesnet/train_timesnet_mse.py```
What to Expect:

This command trains the TimesNet model .using the MSE loss function.
Logs will include epoch-wise loss and validation metrics.
What to Check:

Verify that the MSE-trained TimesNet model checkpoints (e.g., timesnet_nordbyen_mse.pt) are saved in the models directory.
Check the logs for proper training progress.

6. Train TimesNet with Probabilistic Loss
```python timesnet/train_timesnet_nordbyen.py ```
What to Expect:

This command trains the TimesNet model with a probabilistic loss function, enabling uncertainty predictions.
Logs will include probabilistic metrics like quantile loss.
What to Check:

Ensure that the probabilistic model checkpoints (e.g., timesnet_prob.pt) are saved in the models directory.
Confirm that the logs show improving probabilistic metrics.

7. Evaluate NHiTS Models
``` python NHiTS/evaluate_nhits_nordbyen.py --model_path models/nhits_nordbyen.pt  ```
```python NHiTS/evaluate_nhits_nordbyen.py --model_path models/nhits_nordbyen_mse.pt```
```python NHiTS/evaluate_nhits_nordbyen.py --model_path models/nhits_prob.pt```

8. Evaluate TimesNet Models
``` python timesnet/evaluate_timesnet_nordbyen.py --model_path models/timesnet_nordbyen.pt```

```python timesnet/evaluate_timesnet_nordbyen.py --model_path models/timesnet_nordbyen_mse.pt```

```python timesnet/evaluate_timesnet_nordbyen.py --model_path models/timesnet_prob.pt```

9. Run Orchestrator Script
```python run_compare_nhits_timesnet.py```

What to Expect:
This command runs the full pipeline, comparing NHiTS and TimesNet models across all variants (MAE, MSE, Probabilistic).
Logs will summarize the comparison results, including metrics for each model.

What to Check:
Verify that the orchestrator script generates a summary report in the results directory.
Check the logs for any errors and ensure all models are evaluated.

