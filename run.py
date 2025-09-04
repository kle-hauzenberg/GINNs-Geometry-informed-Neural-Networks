"""
Entry point for starting the training.
Checks and performs global config, including the device and the model.
"""
import os
import random
import wandb

from util.get_config import read_cli_args_and_get_config, set_cuda_devices
from GINN.speed.mp_manager import MPManager
import multiprocessing as mp
import logging


def main():

    # read CLI args
    config = read_cli_args_and_get_config()
    set_cuda_devices(config['gpu_list'])

    # set up logging
    log_level = config.get('log_level', 'INFO')
    logging.basicConfig(level=log_level.upper())  # e.g. debug, becomes DEBUG
    logging.info(f'Log level set to {log_level}')
    # explicitly set the log level for other modules
    logging.getLogger('PIL').setLevel(logging.INFO)

    # create process pool before CUDA is initialized
    # either PH or Fenitop need multiprocessing
    mp_manager = MPManager(config.get('num_workers', 0))
    mp_pool_scc = None
    if config['lambda_scc'] > 0:
        # mp_top = MPManager(min(config['ginn_bsize'], config['ph']['ph_max_num_workers']))
        mp_pool_scc = mp.Pool(min(config['ginn_bsize'], config['ph']['ph_max_num_workers']))
    
    mp_top = None
    if config['lambda_comp'] > 0:
        mp_top = (mp.Queue(), mp.Queue())
        
    # initialize torch AFTER multiprocessing pool is created and CUDA_VISIBLE_DEVICES is set
    # also transitive import of torch should be done after CUDA_VISIBLE_DEVICES is set (TODO: check if this is necessary)
    import torch
    from train.ginn_trainer import Trainer
    from util.misc import set_all_seeds

    set_all_seeds(config.get('seed', random.randint(20, 1000000)))

    # Choose device after CUDA visibility has been configured
    if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
        if torch.cuda.device_count() > 1:
            for i_cuda in range(torch.cuda.device_count()):
                print(torch.cuda.get_device_properties(i_cuda).name)
            raise NotImplementedError('Multi-GPU training not supported yet')
        device = 'cuda:0'
        print(f'Visible CUDA devices: {os.getenv("CUDA_VISIBLE_DEVICES")} - using device {device}')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = 'mps'
        print('Using Apple Silicon MPS backend.')
    else:
        device = 'cpu'
        print('CUDA not available - proceeding on CPU')

    # Modern PyTorch way: set default device & dtype (avoid deprecated set_default_tensor_type)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    # Make device visible to the rest of the code if needed
    config['device'] = device
    
    # NOTE: to disable wandb set the ENV
    # "WANDB_MODE": "disabled"
    print(f"WANDB_MODE: {os.getenv('WANDB_MODE')}")
    wandb_save_dir = os.getenv('WANDB_SAVE_DIR', 'wandb')
    print(f'wandb_save_dir: {wandb_save_dir}')
    wandb.init(entity=config['wandb_entity_name'],
               project=config['wandb_project_name'],
               name=config["wandb_experiment_name"],
               dir=wandb_save_dir,
               config=config)
    
    wandb_id = 'no_wandb_id'
    if os.getenv('WANDB_MODE') != 'disabled':
        wandb_id = wandb.run.id
    config['wandb_id'] = wandb_id

    trainer = Trainer(config, mp_manager, mp_pool_scc, mp_top)
    trainer.train()

if __name__ == '__main__':
    main()
