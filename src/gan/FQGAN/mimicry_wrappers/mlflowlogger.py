"""
Logging Wrappers for Mlflow.
https://github.com/ymym3412/Hydra-MLflow-experiment-management/blob/master/mlflow_writer.py
"""
import mlflow
from mlflow import pytorch
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


class MlflowLogger:

    def __init__(self,
                 experiment_name,
                 tracking_uri=None,
                 registry_uri=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.client = MlflowClient(tracking_uri, registry_uri)

        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except MlflowException:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

    def log_params_from_omegaconf_dict(self, params):
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f'{parent_name}.{k}', v)
                else:
                    self.client.log_param(self.run_id, f'{parent_name}.{k}', v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f'{parent_name}.{i}', v)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_torch_model(self, model, model_name):
        with mlflow.start_run(self.run_id):
            pytorch.log_model(model, model_name)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)
