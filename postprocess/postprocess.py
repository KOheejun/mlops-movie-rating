import os

import pandas as pd
from sqlalchemy import create_engine, text


def get_engine(db_name):
    engine = create_engine(
        url=(
            f"mysql+mysqldb://"
            f"{os.environ.get('DB_USER')}:"  # root:
            f"{os.environ.get('DB_PASSWORD')}@"  # root@
            f"{os.environ.get('DB_HOST')}:"  # my-mlops-db:
            f"{os.environ.get('DB_PORT')}/"  # 3306/
            f"{db_name}"  # mlops
        )  # mysql+mysqldb://root:root@my-mlops-db:3306/mlops
    )
    return engine


def write_db(data: pd.DataFrame, db_name, table_name):
    engine = get_engine(db_name)
    with engine.connect() as connect:
        data.to_sql(table_name, connect, if_exists="append")


def read_db(db_name, table_name, k=10):
    engine = get_engine(db_name)
    with engine.connect() as connect:
        result = connect.execute(
            statement=text(
                f"select recommend_content_id from {table_name} "
                f"order by `index` desc limit :k"
            ),
            parameters={"table_name": table_name, "k": k}
        )

    contents = [data[0] for data in result]
    return contents
