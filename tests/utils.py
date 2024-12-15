from dataclasses import asdict
import rootutils
import numpy as np

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.dataset_generate.src.transaction import BaseAppTransactionSampler
from datetime import datetime, timedelta


def generate_local_date(application_date: str,
                        length: int,
                        min_diff_day: int,
                        max_diff_day: int,
                        ):
    current_date = datetime.strptime(application_date, '%Y-%m-%d')
    local_date = [np.datetime64(current_date)]

    for i in range(length - 1):
        diff_days = np.random.randint(
            low=min_diff_day,
            high=max_diff_day + 1
        )
        current_date = current_date + timedelta(days=diff_days)
        local_date.append(
            np.datetime64(current_date)
        )

    return np.array(local_date)[::-1]


def generate_synthetic_data(length: int = 5120,
                            number_mcc: int = 2,
                            min_diff_day: int = 0,
                            max_diff_day: int = 1,
                            num_additional_features: int = 25
                            ):
    local_date = generate_local_date(
        '2020-12-01',
        length=length, min_diff_day=min_diff_day,
        max_diff_day=max_diff_day
    )
    return {
        'amount_rur': np.random.random((length, 1)),
        'mcc_code': np.random.randint(low=0, high=number_mcc, size=(length, 1)),
        'local_date': local_date[::-1],
        **{str(i): np.random.random((length, 1)) for i in range(num_additional_features)}
    }


def generate_app(app_sampler: BaseAppTransactionSampler, app_id: str):
    app = app_sampler.sample()

    new_app = {
        "app_id": app_id,
        "APPLICATION_DATE": "2021-11-29",  # может сломаться?
        "feature_arrays": asdict(app),
    }

    new_app["feature_arrays"]["local_date"] = np.array(
        [np.datetime64("2021-11-12T13:21:52.000000")] * len(app.mcc_code)
    )  # вынести в генератор
    new_app["feature_arrays"]["trans_type"] = np.random.choice(
        [1, 3, 4, 7, 9], len(app.mcc_code)
    )
    new_app["feature_arrays"]["trans_country"] = np.random.choice(
        [15, 217], len(app.mcc_code)
    )
    new_app["feature_arrays"]["trans_curency"] = np.random.choice([110], len(app.mcc_code))
    return new_app
