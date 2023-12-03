import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

# This automatically reads in the configuration
@hydra.main(config_path="../", config_name='config')
def go(config):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = "dummy"

    path_root = hydra.utils.get_original_cwd()
    #path_root = os.getcwd()
    # Move to a temporary directory

    path_step_dummy = os.path.join(path_root, "src", "dummy")
    print(f"{path_step_dummy=}")

    _ = mlflow.run(
                uri=path_step_dummy, #f"{config['main']['components_repository']}/get_data",
                entry_point ="main",
                parameters={
                    "parameter1": 1,
                    "parameter2": 2,
                    "parameter3": "test",
                    
                },
            )


if __name__ == "__main__":
    go()
