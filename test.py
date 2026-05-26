"""
Testing script for TGCMF
"""
from run import TGCMF_run

TGCMF_run(model_name='TGCMF', dataset_name='mosi', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='test', is_training=False)
