import os
import random

import numpy as np


def init_seed():
    np.random.seed(0)
    random.seed(0)


def project_path():
    return os.path.join(
        os.path.dirname(  # /opt/mlops/src/utils
            os.path.abspath(__file__)  # /opt/mlops/src/utils/utils.py
        ),
        "..",  # /opt/mlops/src/utils/..
        ".."   # /opt/mlops/src/utils/../..
    )


def model_dir(model_name):  # if model_name : movie_predictor
    return os.path.join(  # /opt/mlops/models/movie_predictor
        project_path(),
        "models",
        model_name
    )


def auto_increment_run_suffix(name: str, pad=3):
    # movie-predictor-001, movie-predictor-002, movie-predictor-003, ...,
    suffix = name.split("-")[-1]
    next_suffix = str(int(suffix) + 1).zfill(pad)
    return name.replace(suffix, next_suffix)

