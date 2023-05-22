import ta3n_training_script

# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

# 2: Define the search space
hyperparam_search_config = {
    'method': 'random',
    'metric':
    {
        'goal': 'maximize',
        'name': 'val_acc'
    },
    'parameters':
    {
        'epochs': {
            "values": [30, 50]
        },
        "optimizer": {
            "values": ["SGD", "Adam"]
        },
        "fc_dim": {
            "values": [1024, 512, 2048]
        },
        "lr": {
            "values": [3e-2, 3e-3, 1.5e-2, 6e-2]
        },
        "lr-steps": {
            "values": [[15, 25], [10, 20]]
        },
        "weight_decay": {
            "values": [1e-4, 1e-3, 1e-5, 5e-4]
        }
    }
}

dataset_shift_configuration = {
    'method': 'grid',
    'metric':
    {
        'goal': 'maximize',
        'name': 'val_acc'
    },
    'parameters':
    {
        'dataset': {
            "values": [
                {"shift": "D1-D2"},
                {"shift": "D1-D3"},
                {"shift": "D2-D1"},
                {"shift": "D2-D3"},
                {"shift": "D3-D1"},
                {"shift": "D3-D2"},
            ]
        },
        "frame_aggregation": {
            "values": ["avgpool", "trn-m"]
        },
    }
}

if __name__ == "__main__":
    if False:
        ta3n_training_script.main()
    else:
        # 3: Start the sweep
        sweep_id = wandb.sweep(
            sweep=hyperparam_search_config,
            project='mldl23-ego-ta3n',
        )
        wandb.agent(sweep_id, function=ta3n_training_script.main, count=15)
