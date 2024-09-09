"""
Package for managing MLFlow-based MLOps.

KEY QUESTION
TODO: is the architecture such that each run contains one model? or
        can one run contain mulitple models?
"""

import mlflow
import os
import pandas as pd
import argparse

DEFAULT_SETUP:bool = globals().get("DEFAULT_SETUP", False)
DEFAULT_ARTIFACTS:dict = globals().get("DEFAULT_ARTIFACTS", {})

def init_metrics_df()->pd.DataFrame:
    return pd.DataFrame(
        {"script", [],
         "archive", [],
         "descriptor", [],
         "key", [],
         "value", [],
         "date", []
         }
    )

def df_add_metric(df, args, key,value,**kws)->None: 
    """
    df_add_metric - helps with the process of adding metrics
    from the training

    Input
    -----

    df, dataframe
        a dataframe to add metrics

    Returns
    -------
    returns df, DataFrame
    """
    data = {
            "key":        key,
            "value":      value,
            "date":       str(datetime.datetime.now())
           }
    data.update(kws)
    if args:
        args = dict(args)
        data.update(args)
    df.append(data) 
    return None


def log_df_as_artifact(df, file_name: str = "metrics.csv"):
    """Save a DataFrame as a CSV and log it as an artifact."""
    df.to_csv(file_name, index=False)
    mlflow.log_artifact(file_name)
    os.remove(file_name)  # Clean up the local file after logging


def describe_run(args: dict,
                 exp_name: str | None = None,
                 run_name: str = "",
                 setup: bool = DEFAULT_SETUP,
                 artifact_kws: dict = DEFAULT_ARTIFACTS) -> tuple:
    """
    `describe_run` takes a dict of args e.g. from (args:Namespace).__dict__
    and it generates the name for the run within the experiment.
    It also starts an MLflow run and logs parameters if setup is Truejj

    Args:
        args (dict): Dictionary of arguments
        name (str): Optional name for the run
        exp (str | None): Optional experiment name
        setup (bool): Whether to start an MLflow run and log parameters
        artifact_kws (dict): Optional artifacts to log

    Returns:
        tuple: (experiment_name, run_name)
    """
    if isinstance(args, argparse.Namespace):
        args = args.__dict__
    exp_name = exp_name or args.get('experiment_name') or args.get('script')
    if not exp_name:
        raise ValueError("Must provide experiment name in exp var, args['experiment_name'], or args['script']")
    if isinstance(exp_name, Exception):
        raise exp_name
    run_name = run_name or f"{args.get('model_type', 'unknown_model')}_{args.get('model_version', 'v1')}"
    for key, val in args.items():
        if key not in ['script', 'experiment_name', 'model_type', 'model_version']:
            run_name += f"_{key}={val}"

    if setup:
        mlflow.set_experiment(exp_name)
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(args)
        if artifact_kws:
            for key, path in artifact_kws.items():
                mlflow.log_artifact(path, key)

    return exp_name, run_name

