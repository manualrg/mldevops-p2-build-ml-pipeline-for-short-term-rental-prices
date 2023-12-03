#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(f"{args.input_artifact}").file()
    df = pd.read_csv(artifact_local_path)

    # Dataprep
    df ['price'] = df['price'].clip(lower=args.min_price, upper=args.max_price)
    df['last_review'] = pd.to_datetime(df['last_review'])

    n_rows_raw = len(df)
    idx = df['longitude'].between(args.min_lon, args.max_lon) & df['latitude'].between(args.min_lat, args.max_lat)
    df = df[idx].copy()
    n_rows_removed_geoloc = n_rows_raw - len(df)
    logger.info("Removed %s rows due to incorrect geolocalization data", n_rows_removed_geoloc)
    
    # Log output artifact
    path_data = os.getenv("PATH_DATA", "data")
    filename = f"{os.path.join(path_data, args.output_artifact)}"
    df.to_csv(filename)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)
    run.log_artifact(artifact)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Raw dataset. Add WNB artifact version, e.g. latest",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Cleaned dataset. Add file extension, e.g. .csv",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Add an artifact type to WNB",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Add an artifact description to WNB",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Floor threshold ($)",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Cap threshold ($)",
        required=True
    )

    parser.add_argument(
        "--min_lat", 
        type=float,
        help="Minimum latitude for NYC (deg)",
        required=True
    )

    parser.add_argument(
        "--max_lat", 
        type=float,
        help="Maximum latitude for NYC (deg)",
        required=True
    )

    parser.add_argument(
        "--min_lon", 
        type=float,
        help="Minimum longitude for NYC (deg)",
        required=True
    )

    parser.add_argument(
        "--max_lon", 
        type=float,
        help="Maximum longitude for NYC (deg)",
        required=True
    )

    args = parser.parse_args()

    go(args)
    quit()
