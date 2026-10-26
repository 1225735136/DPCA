import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os.path
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar

from config.config import get_parser, get_cfg
from data.nuscenes_dataset import NuScenesDataModule
from data.robotcar.robotcar_dataset import RobotCarDataModule
from trainer import Md4All

if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    if cfg.SYSTEM.CPU_THREADS is not None:
        threads = cfg.SYSTEM.CPU_THREADS
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise ValueError('SYSTEM.CPU_THREADS must be a positive integer or null.')
        os.environ['OMP_NUM_THREADS'] = str(threads)
        os.environ['MKL_NUM_THREADS'] = str(threads)
        torch.set_num_threads(threads)

    if cfg.EVALUATION.SAVE.QUANTITATIVE_RES_PATH is not None:
        os.makedirs(cfg.EVALUATION.SAVE.QUANTITATIVE_RES_PATH, exist_ok=True)
    if cfg.EVALUATION.SAVE.QUALITATIVE_RES_PATH is not None:
        os.makedirs(cfg.EVALUATION.SAVE.QUALITATIVE_RES_PATH, exist_ok=True)

    if cfg.DATASET.NAME == "nuscenes":
        dm = NuScenesDataModule(cfg)
    elif cfg.DATASET.NAME == "robotcar":
        dm = RobotCarDataModule(cfg)
    else:
        raise NotImplementedError(f"The dataset {cfg.DATASET.NAME} is not implemented.")

    model = Md4All(cfg, is_train=False)

    if cfg.LOAD.CHECKPOINT_PATH is None:
        raise AssertionError("For evaluation you need to specify a path to a checkpoint")

    trainer = pl.Trainer(
        accelerator=cfg.SYSTEM.ACCELERATOR,
        devices=cfg.SYSTEM.DEVICES,
        precision=cfg.SYSTEM.PRECISION,
        logger=False,
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        enable_progress_bar=True,
        deterministic=cfg.SYSTEM.DETERMINISTIC or cfg.SYSTEM.DETERMINISTIC_ALGORITHMS,
        benchmark=cfg.SYSTEM.BENCHMARK
    )

    # Enable warn_only since some modules do not have a deterministic algorithm implemented
    if cfg.SYSTEM.DETERMINISTIC or cfg.SYSTEM.DETERMINISTIC_ALGORITHMS:
        torch.use_deterministic_algorithms(mode=True, warn_only=True)

    trainer.test(model, dm,
                 ckpt_path=cfg.LOAD.CHECKPOINT_PATH, 
                 weights_only=False)
