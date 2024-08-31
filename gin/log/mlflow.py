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

DEFAULT_SETUP:bool = True
DEFAULT_ARTIFACTS:dict = {}

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


def describe_run(args:dict,
                 name:str="", exp:str|None=None,
                 setup:bool=DEFAULT_SETUP,
                 artifact_kws:dict=DEFAULT_ARTIFACTS):
    """
    `describe_run` takes a dict of args e.g. from (args:Namespace).__dict__
    and it generates the name for the run within the experiment


    """
    # TODO: include option to automatically log these params at the time
    #       of the function call. And we might even consider convenience
    #       arguments to pass on artifacts and metrics.
    if isinstance(args, argparse.Namespace):
        args = args.__dict__
    exp_name = (exp or args['script'] if 'script' in args 
                else ValueError("Must provide script name in exp var ot args"))
    if isinstance(exp_name, Exception):
        raise exp_name
    # And from here, let's crreate the run name
    run_name = name
    for key,val in args.items():
        run_name = f"run_name_key={str(key)}-val={str(val)}"
    if setup:
        mlflow.start_run(run_name)
        mlflow.log_params(args)
        if artifact_kws: mlflow.log_artifacts(artifact_kws)
    return exp_name, run_name
