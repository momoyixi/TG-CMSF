"""
Training script for TGCMSF
"""
from run import TGCMSF_run

TGCMSF_run(model_name='TGCMSF', dataset_name='mosi', is_tune=False, seeds=[1111], 
# model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train', is_training=True)
