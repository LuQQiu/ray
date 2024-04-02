from functools import wraps
import json
import multiprocessing
from multiprocessing import Process
import numpy as np
import os
import pandas as pd
import time
import traceback
from typing import Dict
import xgboost as xgb

import ray
from ray import data
from ray.train.xgboost import XGBoostTrainer
from ray.train import RunConfig, ScalingConfig

import fsspec
from alluxiofs import AlluxioFileSystem

_XGB_MODEL_PATH = "model.json"
_TRAINING_TIME_THRESHOLD = 1000
_PREDICTION_TIME_THRESHOLD = 450

_EXPERIMENT_PARAMS = {
    "smoke_test": {
        "data": "s3://ai-ref-arch/10G-xgboost-data/d9ef953e9a7347db8793f9e772357e68_000076.parquet",
        "num_workers": 1,
        "cpus_per_worker": 1,
    },
    "10G": {
        "data": "s3://ai-ref-arch/10G-xgboost-data/",
        "num_workers": 1,
        "cpus_per_worker": 12,
    },
    "100G": {
        "data": "s3://ai-ref-arch/100G-xgboost-data.parquet/",
        "num_workers": 10,
        "cpus_per_worker": 12,
    },
}


def run_and_time_it(f):
    """Runs f in a separate process and times it."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        class MyProcess(Process):
            def __init__(self, *args, **kwargs):
                super(MyProcess, self).__init__(*args, **kwargs)
                self._pconn, self._cconn = multiprocessing.Pipe()
                self._exception = None

            def run(self):
                try:
                    super(MyProcess, self).run()
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    self._cconn.send(e)

            @property
            def exception(self):
                if self._pconn.poll():
                    self._exception = self._pconn.recv()
                return self._exception

        p = MyProcess(target=f, args=args, kwargs=kwargs)
        start = time.monotonic()
        p.start()
        p.join()
        if p.exception:
            raise p.exception
        time_taken = time.monotonic() - start
        print(f"{f.__name__} takes {time_taken} seconds.")
        return time_taken

    return wrapper


def setup_alluxio(args):
    fsspec.register_implementation("alluxio", AlluxioFileSystem, clobber=True)
    alluxio_kwargs = {}
    if args.alluxio_etcd_hosts and args.alluxio_worker_hosts:
        raise ValueError("Either etcd_hosts or worker_hosts should be provided, not both.")
    if args.alluxio_etcd_hosts:
        alluxio_kwargs['etcd_hosts'] = args.alluxio_etcd_hosts
    if args.alluxio_worker_hosts:
        alluxio_kwargs['worker_hosts'] = args.alluxio_worker_hosts
    alluxio_kwargs['target_protocol'] = "s3"

    alluxio_options = {}
    if args.alluxio_page_size:
        alluxio_options['alluxio.worker.page.store.page.size'] = args.alluxio_page_size
    if args.alluxio_cluster_name:
        alluxio_options['alluxio.cluster.name'] = args.alluxio_cluster_name
    if alluxio_options:
        alluxio_kwargs['options'] = alluxio_options
    return fsspec.filesystem("alluxio", **alluxio_kwargs)

@run_and_time_it
def run_xgboost_training(data_path: str, num_workers: int, cpus_per_worker: int, args):
    if args.use_alluxio:
        alluxio = setup_alluxio(args)
        ds = data.read_parquet(data_path, filesystem = alluxio)
    else: 
        ds = data.read_parquet(data_path)
    # for row in ds.iter_rows():
    #     pass
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    }

    trainer = XGBoostTrainer(
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            resources_per_worker={"CPU": cpus_per_worker} if cpus_per_worker is not None else {},
        ),
        label_column="labels",
        params=params,
        datasets={"train": ds},
        run_config=RunConfig(
            storage_path="/tmp/ray_tmp_storage", name="xgboost_benchmark"
        ),
    )
    result = trainer.fit()
    xgboost_model = XGBoostTrainer.get_model(result.checkpoint)
    xgboost_model.save_model(_XGB_MODEL_PATH)
    ray.shutdown()


@run_and_time_it
def run_xgboost_prediction(model_path: str, data_path: str):
    model = xgb.Booster()
    model.load_model(model_path)
    ds = data.read_parquet(data_path)
    ds = ds.drop_columns(["labels"])

    class XGBoostPredictor:
        def __init__(self, model: xgb.Booster):
            self.model = model

        def __call__(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
            dmatrix = xgb.DMatrix(data)
            return {"predictions": self.model.predict(dmatrix)}

    concurrency = int(ray.cluster_resources()["CPU"] // 2)
    result = ds.map_batches(
        XGBoostPredictor,
        # Improve prediction throughput for xgboost with larger
        # batch size than default 4096
        batch_size=8192,
        concurrency=concurrency,
        fn_constructor_kwargs={"model": model},
        batch_format="pandas",
    )

    for _ in result.iter_batches():
        pass

    return result


def main(args):
    experiment = args.size if not args.smoke_test else "smoke_test"
    experiment_params = _EXPERIMENT_PARAMS[experiment]

    data_path = experiment_params["data"]
    num_workers = args.overwrite_num_workers if args.overwrite_num_workers is not None else experiment_params["num_workers"]
    cpus_per_worker = args.overwrite_cpus_per_worker if args.overwrite_cpus_per_worker is not None else experiment_params["cpus_per_worker"]
    if cpus_per_worker == -1:
        cpus_per_worker = None  # Or any default logic you prefer when "-1" is passed

    print("Running xgboost training benchmark...")
    training_time = run_xgboost_training(data_path, num_workers, cpus_per_worker, args)
    print("Running xgboost prediction benchmark...")
    # prediction_time = run_xgboost_prediction(_XGB_MODEL_PATH, data_path)
    result = {
        "training_time": training_time,
        # "prediction_time": prediction_time,
    }
    print("Results:", result)
    # test_output_json = os.environ.get("TEST_OUTPUT_JSON", "/tmp/result.json")
    # with open(test_output_json, "wt") as f:
    #     json.dump(result, f)

    # if not args.disable_check:
    #     if training_time > _TRAINING_TIME_THRESHOLD:
    #         raise RuntimeError(
    #             f"Training on XGBoost is taking {training_time} seconds, "
    #             f"which is longer than expected ({_TRAINING_TIME_THRESHOLD} seconds)."
    #         )

    #     if prediction_time > _PREDICTION_TIME_THRESHOLD:
    #         raise RuntimeError(
    #             f"Batch prediction on XGBoost is taking {prediction_time} seconds, "
    #             f"which is longer than expected ({_PREDICTION_TIME_THRESHOLD} seconds)."
    #         )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, choices=["10G", "100G"], default="100G")
    # Add a flag for disabling the timeout error.
    # Use case: running the benchmark as a documented example, in infra settings
    # different from the formal benchmark's EC2 setup.
    parser.add_argument(
        "--disable-check",
        action="store_true",
        help="disable runtime error on benchmark timeout",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--use-alluxio",
        action="store_true",
        default=False,
        help="Whether to use Alluxio instead of original ufs filesystem for data loading.",
    )
    parser.add_argument(
        "--alluxio-etcd-hosts",
        default=None,
        help="The ETCD host to connect to to get Alluxio workers connection info.",
    )
    parser.add_argument(
        "--alluxio-worker-hosts",
        default=None,
        help="The worker hostnames in host1,host2,host3 format. Either etcd_host or worker_hosts should be provided, not both.",
    )
    parser.add_argument(
        "--alluxio-page-size",
        default=None,
        help="The alluxio page size of Alluxio servers.",
    )
    parser.add_argument(
        "--alluxio-cluster-name",
        default=None,
        help="The alluxio cluster name of the Alluxio servers.",
    )
    parser.add_argument(
        "--overwrite-num-workers",
        type=int,
        default=4,
        help="Overwrite the number of workers.",
    )
    parser.add_argument(
        "--overwrite-cpus-per-worker",
        type=int,
        default=3,
        help="Overwrite cpus per worker. -1 means do not set it.",
    )
    args = parser.parse_args()
    main(args)
