import os
import sys
from time import sleep

sys.path.append(
    os.path.dirname(  # /opt/mlops
        os.path.dirname(  # /opt/mlops/src
            os.path.abspath(__file__)  # /opt/mlops/src/main.py
        )
    ),
)

import fire
import wandb
import numpy as np
from icecream import ic
from tqdm import tqdm
from dotenv import load_dotenv

from src.dataset.watch_log import get_datasets
from src.dataset.data_loader import SimpleDataLoader
from src.model.movie_predictor import MoviePredictor, model_save
from src.utils.utils import init_seed, auto_increment_run_suffix
from src.utils.enums import ModelTypes
from src.train.train import train
from src.evaluate.evaluate import evaluate
from src.inference.inference import (
    load_checkpoint, init_model, inference, recommend_to_df
)
from src.postprocess.postprocess import write_db


init_seed()
load_dotenv()


def get_runs(project_name):
    return wandb.Api().runs(path=project_name, order="-created_at")


def get_latest_run(project_name):
    runs = get_runs(project_name)
    if not runs:
        return f"{project_name}-000"  # auto_increment_run_suffix(movie-predictor-000) => movie-predictor-001

    return runs[0].name


def run_preprocessing(start_date, end_date):
    ic(start_date, end_date)
    print("run preprocessing")


def run_train(model_name, num_epochs=10, batch_size=64):
    """
    run train.

    Example:
        python main.py train --model_name movie_predictor -n 30
    """
    ModelTypes.validation(model_name)

    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)
    
    project_name = model_name.replace("_", "-")  # movie_predictor => movie-predictor
    run_name = get_latest_run(project_name)
    next_run_name = auto_increment_run_suffix(run_name)

    wandb.init(
        project=project_name,
        id=next_run_name,
        name=next_run_name,
        notes="content-based movie recommend model",
        tags=["content-based", "movie", "recommend"],
        config=locals(),
    )

    train_dataset, val_dataset, test_dataset = get_datasets()
    train_loader = SimpleDataLoader(train_dataset.features, train_dataset.labels, batch_size=batch_size, shuffle=True)
    val_loader = SimpleDataLoader(val_dataset.features, val_dataset.labels, batch_size=batch_size, shuffle=True)
    test_loader = SimpleDataLoader(test_dataset.features, test_dataset.labels, batch_size=batch_size, shuffle=True)

    model_params = {
        "input_dim": train_dataset.features_dim,
        "num_classes": train_dataset.num_classes,
        "hidden_dim": 64
    }

    model_class = ModelTypes[model_name.upper()].value
    model = model_class(**model_params)

    # num_epochs = 10
    for epoch in tqdm(range(num_epochs)):
        train_loss = train(model, train_loader)
        val_loss, _ = evaluate(model, val_loader)
        
        #sleep(1)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val-Train Loss : {val_loss-train_loss:.4f}")
        wandb.log({"Loss/Train": train_loss})
        wandb.log({"Loss/Valid": val_loss})

    wandb.finish()

    test_loss, predictions = evaluate(model, test_loader)
    #print(f"{test_loss=:.4f}")
    ic(test_loss)
    #print([train_dataset.decode_content_id(idx) for idx in predictions])

    model_save(
        model=model,
        model_params=model_params,
        epoch=num_epochs,
        loss=train_loss,
        scaler=train_dataset.scaler,
        label_encoder=train_dataset.label_encoder,
    )


def run_inference(data=None, batch_size=64):
    checkpoint = load_checkpoint()
    model, scaler, label_encoder = init_model(checkpoint)
    
    if data is None:
        data = []

    data = np.array(data)
    
    recommend = inference(model, scaler, label_encoder, data, batch_size)
    print(recommend)

    recommend_df = recommend_to_df(recommend)
    write_db(recommend_df, "mlops", "recommend")


if __name__ == "__main__":
    fire.Fire({
        "preprocessing": run_preprocessing,
        "train": run_train,
        "inference": run_inference,
    })
