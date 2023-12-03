#!/usr/bin/env python
"""
Performs dummy testing and save the results in Weights & Biases
"""
import sys
import os
import argparse
import logging
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="dummy")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    print(f"Py version: {sys.version}")
    print(f"Py executable: {sys.executable}")
    print(f"CWD: {os.getcwd()}")
    print(f"Args: {args}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step test the environment")


    parser.add_argument(
        "--parameter1", 
        type=int,
        help="parameter1",
        required=True
    )

    parser.add_argument(
        "--parameter2", 
        type=int,
        help="parameter2",
        required=True
    )

    parser.add_argument(
        "--parameter3", 
        type=str,
        help="parameter3",
        required=True
    )


    args = parser.parse_args()

    go(args)
    logger.info("SUCCESS")
    quit()
