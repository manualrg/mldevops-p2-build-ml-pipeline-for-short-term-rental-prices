import json
import logging
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]

PATH_DATA = "data"

def check_path_data(path):

    if not os.path.exists(path):
        logger.info("Making path: %s", )
        os.mkdir(path)


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # data folder on each component
    os.environ["PATH_DATA"] = PATH_DATA

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps


    path_root = hydra.utils.get_original_cwd()


    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            # conda file from provided repo is not working, changed to source directory to modify conda
            path_step_download = os.path.join(path_root, "components", "get_data")
            _ = mlflow.run(
                path_step_download, #f"{config['main']['components_repository']}/get_data",
                "main",
                #version='main',
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )
            logging.info("SUCCESS: Finished download step")

        if "basic_cleaning" in active_steps:
            # perform data cleaning
            path_step_basic_cleaning = os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning")
            check_path_data(os.path.join(path_step_basic_cleaning, PATH_DATA))
            _ = mlflow.run(
                path_step_basic_cleaning,
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price'],
                    "min_lat": config['etl']['min_lat'],
                    "max_lat": config['etl']['max_lat'],
                    "min_lon": config['etl']['min_lon'],
                    "max_lon": config['etl']['max_lon']
                },
            )
            logging.info("SUCCESS: Finished basic_cleaning step")

        if "data_check" in active_steps:
            # Run assertions on input data
            path_step_data_check = os.path.join(hydra.utils.get_original_cwd(), "src", "data_check")

            _ = mlflow.run(
                path_step_data_check,
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config['data_check']['kl_threshold'],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price'],
                    "min_lat": config['etl']['min_lat'],
                    "max_lat": config['etl']['max_lat'],
                    "min_lon": config['etl']['min_lon'],
                    "max_lon": config['etl']['max_lon']
                },
            )
            logging.info("SUCCESS: Finished data_check step")

        if "data_split" in active_steps:
            # Splits data in two separated subsetes: trainval.csv and test.csv
            # conda file from provided repo is not working, changed to source directory to modify conda
            path_step_data_split = os.path.join(path_root, "components", "train_val_test_split")
            _ = mlflow.run(
                path_step_data_split, #f"{config['main']['components_repository']}/data_split",
                "main",
                #version='main',
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )
            logging.info("SUCCESS: Finished data_split step")

        if "train_random_forest" in active_steps:
            # train a randon forest model on trainval_data

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            path_step_train_random_forest = os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest")

            _ = mlflow.run(
                path_step_train_random_forest,
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "rf_config": rf_config,
                    "max_tfidf_features": config['modeling']['max_tfidf_features'],
                    "output_artifact": "random_forest_export"
                },
            )
            logging.info("SUCCESS: Finished train_random_forest step")

        if "test_regression_model" in active_steps:
            # Test the model tagged as "prod" in WNB on a holdout dataset (test_data.csv)
            # This step needs to be triggered manually after tagging a model as "production ready"
            # conda file from provided repo is not working, changed to source directory to modify conda
            path_step_test_regression_model = os.path.join(hydra.utils.get_original_cwd(), "components", "test_regression_model")

            _ = mlflow.run(
                path_step_test_regression_model,
                "main",
                parameters={
                    "mlflow_model": "random_forest_export:prod",
                    "test_dataset": "test_data.csv:latest"
                },
            )
            logging.info("SUCCESS: Finished test_regression_model step")


if __name__ == "__main__":
    go()
