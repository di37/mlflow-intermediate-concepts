# Experiment Tracking with MLFlow

## Setting up Environment

- Create a new environment:

```bash
conda create -n ml_experiment_mlflow python=3.9
```

- Activate the environment:

```bash
conda activate ml_experiment_mlflow
```

- Install the requirements

```bash
pip install -r requirements.txt
```

- Export the environment

```bash
conda env export --no-builds > environment.yml
```

## Running MLFlow UI

- Spin up MLFlow server locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

In general:

```bash
mlflow ui --backend-store-uri [db_uri]
```

- Fixes if MLFlow is not working:

```bash
mlflow db upgrade sqlite:///mlflow.db
```

In general:

```bash
mlflow db upgrade [db_uri]
```

## Experiment Tracking

- Follow the runs in the notebook for training the ml model and hyperparameter tuning.
- Once we run couple of experiments, we are then ready to compare and contrast in order to find the optimal hyperparameters vs. evaluation metrics of the models and accordingly choose the best model.
- We will be able to visualize all sorts of plots we want.
- Once we obtain optimal model parameters, we will train using these params.

Summary:

- Hyperparameter Tuning of XGBoost is done using hyperopt library.
- Then we make use of MLFlow to visualize the parameters and evaluation metrics.
- At the end, we also see `autolog` in mlflow that allows to capture most of the params, models, and other important information.

```python
mlflow.xgboost.autolog()
```

## Model Management

Log models as an artifact:

```python
mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")
```

Logging the models in MLFlow:

```python
mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")
```

In general:

```python
mlflow.<framework>.log_model(model, artifact_path="<models_path>")
```

Once we logged the preprocessor and model, now it is ready for inference. We can copy paste from the artifacts in MLFlow UI. Check the notebook - `02_duration_prediction_model_management.ipynb`.

## Model Registry

- Model can be either pushed to current stage: Staging, Production and Archived.
- Models can be either registered through the ui or python commands using mlflow library.
- Check the notebook - `03_duration_prediction_model_registry.ipynb` for the model registry using the mlflow library command lines.
