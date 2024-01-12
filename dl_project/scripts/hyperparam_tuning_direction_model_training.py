import sys
import os
import torch
import yaml
from pathlib import Path
from omegaconf import OmegaConf

from ax.service.ax_client import AxClient, ObjectiveProperties

sys.path.append(str(Path(__file__).parent.parent))

from dl_project.scripts.train_direction_model import DirectionModelTrainer
from dl_project.scripts.validate_direction_model import DirectionModelValidator

class DirectionModelTuning:

    def __init__(self, exp_params, training_conf, tuning_conf):
        self.__training_conf = training_conf
        self.__ax_client = AxClient()
        self.__ax_client.create_experiment(
            name="direction_model_tuning",
            parameters=exp_params,
            objectives={"loss": ObjectiveProperties(minimize=True)}
        )
        self.__training_epochs = tuning_conf["epochs"]
        self.__training_conf["model"]["direction_model"]["path"] = tuning_conf["model_state_path"]
        self.__training_conf["epochs"] = self.__training_epochs
        
    
    def get_best_parameters(self):
        best_parameters, values = self.__ax_client.get_best_parameters()
        output = {
            "best_parameters": best_parameters,
            "mean": values[0],
            "variance": values[1]
        }
        return output
    
    def run_tuning(self, total_trials):
        for i in range(total_trials):
            parameters, trial_index = self.__ax_client.get_next_trial()
            # Local evaluation here can be replaced with deployment to external system.
            self.__ax_client.complete_trial(trial_index=trial_index, raw_data=self.train_evaluate(parameters))
    
    def save_tuning_state(self, path):
        self.__ax_client.save_to_json_file(path)
    
    def load_tuning_state(self, path):
        self.__ax_client = self.__ax_client.load_from_json_file(path)
    
    def train_evaluate(self, parameters):

        self.__training_conf["lr"] = int(parameters["lr"])
        self.__training_conf["model"]["direction_model"]["depth"] = int(parameters["depth"])
        self.__training_conf["model"]["direction_model"]["alpha"] = parameters["alpha"]
        self.__training_conf["strength"] = parameters["strength"]

        training = DirectionModelTrainer(configs=self.__training_conf)
        validation = DirectionModelValidator(configs=self.__training_conf)
        training.train()
        validation_loss = validation.validate()

        return {
            "loss": validation_loss
        }
        
    
    def __attach_trials(self, trials):
        for i, trial in enumerate(trials):
            self.__ax_client.attach_trial(
                parameters=trial
            )

            # Get the parameters and run the trial 
            baseline_parameters = self.__ax_client.get_trial_parameters(trial_index=i)
            self.__ax_client.complete_trial(trial_index=i, raw_data=self.train_evaluate(baseline_parameters))

    

if __name__ == '__main__':
    training_conf_path = Path('dl_project/configs/direction_model_training.yaml')
    tuning_conf_path = Path('dl_project/configs/direction_model_tuning.yaml')


    tuning_conf = OmegaConf.load(tuning_conf_path)['tuning']
    training_conf = OmegaConf.load(training_conf_path)['training']

    tuning_state_path = Path(tuning_conf['tuning_state_path'])

    exp_params = [
                {
                    "name": "depth",
                    "type": "range",
                    "bounds": [1, 2],
                    "log_scale": False
                },
                {
                    "name": "alpha",
                    "type": "range",
                    "bounds": [0.001, 0.5],
                    "log_scale": False
                },
                {
                    "name": "lr",
                    "type": "range",
                    "bounds": [0.000001, 0.1],
                    "log_scale": True
                },
                {
                    "name": "strength",
                    "type": "range",
                    "bounds": [0.01, 0.3],
                    "log_scale": False
                }
                
    ]

    tuning = DirectionModelTuning(exp_params, training_conf, tuning_conf)
    if os.path.exists(tuning_state_path):
        tuning.load_tuning_state(tuning_state_path)
    tuning.run_tuning(10)
    tuning.save_tuning_state(tuning_state_path)
    print(tuning.get_best_parameters())

