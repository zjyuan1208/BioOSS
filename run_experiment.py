"""
This script loads hyperparameters from JSON files and trains models on specified datasets using
the `create_dataset_model_and_train` function from `train.py` or its PyTorch equivalent. The results
are saved in the output directories defined in the JSON files.

The `run_experiments` function iterates over model names and dataset names, loading configuration
files from a specified folder, and then calls the appropriate training function based on the
framework (PyTorch or JAX).

Arguments for `run_experiments`:
- `model_names`: List of model architectures to use.
- `dataset_names`: List of datasets to train on.
- `experiment_folder`: Directory containing JSON configuration files.
- `pytorch_experiments`: Boolean indicating whether to use PyTorch (True) or JAX (False).

The script also provides a command-line interface (CLI) for specifying whether to run PyTorch experiments.

Usage:
- Use the `--pytorch_experiments` flag to run experiments with PyTorch; otherwise, JAX is used by default.
"""

import argparse
import json
import diffrax
import os
from train import create_dataset_model_and_train

def run_experiments(model_names, dataset_names, experiment_folder):
    for model_name in model_names:
        for dataset_name in dataset_names:
            with open(
                    experiment_folder + f"/{model_name}/{dataset_name}.json", "r"
            ) as file:
                data = json.load(file)

            seeds = data["seeds"]
            # data_dir = data["data_dir"]
            data_dir = '/home/zhyuan/Desktop/FTDToss/data_dir'
            output_parent_dir = data["output_parent_dir"]
            lr_scheduler = eval(data["lr_scheduler"])
            num_steps = data["num_steps"]
            print_steps = data["print_steps"]
            batch_size = data["batch_size"]
            metric = data["metric"]

            # Handle model-specific parameters
            if model_name == 'LinOSS':
                linoss_discretization = data["linoss_discretization"]
            else:
                linoss_discretization = None

            # Add FDTD-specific parameters
            if model_name == 'FDTD':
                # Default to None if not present in config
                fdtd_wave_speed = data.get("fdtd_wave_speed", 1.0)
                fdtd_damping_p = data.get("fdtd_damping_p", 0.0001)
                fdtd_damping_v = data.get("fdtd_damping_v", 0.0001)
            else:
                fdtd_wave_speed = None
                fdtd_damping_p = None
                fdtd_damping_v = None

            use_presplit = data["use_presplit"]
            T = data["T"]

            # Determine if model uses dt0 parameter
            if model_name in ["lru", "S5", "S6", "mamba", "LinOSS", "FDTD"]:
                dt0 = None
            else:
                dt0 = float(data["dt0"])

            scale = data["scale"]
            lr = float(data["lr"])
            include_time = data["time"].lower() == "true"
            hidden_dim = int(data["hidden_dim"])

            # Handle model-specific configurations
            if model_name in ["log_ncde", "nrde", "ncde"]:
                vf_depth = int(data["vf_depth"])
                vf_width = int(data["vf_width"])
                if model_name in ["log_ncde", "nrde"]:
                    logsig_depth = int(data["depth"])
                    stepsize = int(float(data["stepsize"]))
                else:
                    logsig_depth = 1
                    stepsize = 1
                if model_name == "log_ncde":
                    lambd = float(data["lambd"])
                else:
                    lambd = None
                ssm_dim = None
                num_blocks = None
            else:
                vf_depth = None
                vf_width = None
                logsig_depth = 1
                stepsize = 1
                lambd = None
                ssm_dim = int(data["ssm_dim"])
                num_blocks = int(data["num_blocks"])

            # Additional parameters for specific models
            if model_name in ["S5", "LinOSS", "FDTD"]:
                ssm_blocks = int(data["ssm_blocks"])
            else:
                ssm_blocks = None

            # Handle dataset-specific parameters
            if dataset_name == "ppg":
                output_step = int(data["output_step"])
            else:
                output_step = 1

            # Prepare model arguments
            model_args = {
                "num_blocks": num_blocks,
                "hidden_dim": hidden_dim,
                "vf_depth": vf_depth,
                "vf_width": vf_width,
                "ssm_dim": ssm_dim,
                "ssm_blocks": ssm_blocks,
                "dt0": dt0,
                "solver": diffrax.Heun(),
                "stepsize_controller": diffrax.ConstantStepSize(),
                "scale": scale,
                "lambd": lambd,
                # Add FDTD-specific parameters
                "fdtd_wave_speed": fdtd_wave_speed,
                "fdtd_damping_p": fdtd_damping_p,
                "fdtd_damping_v": fdtd_damping_v,
            }

            # Prepare run arguments
            run_args = {
                "data_dir": data_dir,
                "use_presplit": use_presplit,
                "dataset_name": dataset_name,
                "output_step": output_step,
                "metric": metric,
                "include_time": include_time,
                "T": T,
                "model_name": model_name,
                "stepsize": stepsize,
                "logsig_depth": logsig_depth,
                "linoss_discretization": linoss_discretization,
                "model_args": model_args,
                "num_steps": num_steps,
                "print_steps": print_steps,
                "lr": lr,
                "lr_scheduler": lr_scheduler,
                "batch_size": batch_size,
                "output_parent_dir": output_parent_dir,
                "id": id,
            }
            run_fn = create_dataset_model_and_train

            for seed in seeds:
                print(f"Running experiment with seed: {seed}")
                run_fn(seed=seed, **run_args)


# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
#     args = argparse.ArgumentParser()
#     args.add_argument("--dataset_name", type=str, default='SelfRegulationSCP1',
#                       help="'EigenWorms','EthanolConcentration', 'Heartbeat', "
#                            "'MotorImagery','SelfRegulationSCP1', 'SelfRegulationSCP2','ppg'")
#     args.add_argument("--model_name", type=str, default='LinOSS',
#                       help="Model to use: 'LinOSS', 'FDTD', 'log_ncde', 'lru', 'ncde', 'nrde', 'S5'")
#     args = args.parse_args()
#
#     model_names = [args.model_name]
#     dataset_names = [
#         args.dataset_name
#     ]
#     experiment_folder = "experiment_configs/repeats"
#     # experiment_folder = "experiment_configs/test"
#
#     run_experiments(model_names, dataset_names, experiment_folder)

def get_all_models_and_datasets():
    """
    Returns the predefined lists of models and datasets.

    Returns:
        Tuple of (list of models, list of datasets)
    """
    # Predefined list of models
    model_names = [
        # 'lru',
        # 'S5'
        'LinOSS'
        # 'FDTD'
    ]
    # 'LinOSS', 'log_ncde', 'lru'

    # Predefined list of datasets
    dataset_names = [
        # 'EthanolConcentration',
        # 'Heartbeat',
        # 'EigenWorms',
        # 'MotorImagery',
        # 'SelfRegulationSCP1',
        # 'SelfRegulationSCP2',
        'ppg'
    ]

    return model_names, dataset_names


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run experiments for all models and datasets")
    parser.add_argument("--experiment_folder", type=str, default="experiment_configs/repeats",
                        help="Directory containing experiment configurations")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device to use (CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Get all models and datasets
    model_names, dataset_names = get_all_models_and_datasets()

    print(f"Found {len(model_names)} models: {', '.join(model_names)}")
    print(f"Found {len(dataset_names)} datasets: {', '.join(dataset_names)}")

    # Confirm with user
    print(f"\nAbout to run {len(model_names) * len(dataset_names)} model-dataset combinations.")
    confirmation = input("Continue? (y/n): ")
    if confirmation.lower() != 'y':
        print("Aborting.")
        exit(0)

    # Run experiments
    run_experiments(model_names, dataset_names, args.experiment_folder)