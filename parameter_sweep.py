import ta3n_training_script

# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

# 2: Define the search space
hyperparam_search_config = {
    'method': 'grid',
    'metric':
    {
        'goal': 'maximize',
        'name': 'best_acc'
    },
    'parameters':
    {
        'epochs': {
            "values": [30]
        },
        'dataset': {
            "values": [
                {"shift": "D1-D3"},
                {"shift": "D2-D3"},
                {"shift": "D3-D2"},
            ]
        },
        "fc_dim": {
            "values": [1024]
        },
        "lr": {
            "values": [6e-2]
        },
        "lr-adaptive": {
            "values": ["dann"]
        },
        "weight_decay": {
            "values": [1e-4]
        }
    }
}

dataset_shift_configuration = {
    'method': 'grid',
    'metric':
    {
        'goal': 'maximize',
        'name': 'best_acc'
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
        'epochs': {
            "values": [30]
        },
        "fc_dim": {
            "values": [1024]
        },
        "lr": {
            "values": [6e-2]
        },
        "lr-adaptive": {
            "values": ["dann"]
        },
        "weight_decay": {
            "values": [1e-4]
        }
    }
}

if __name__ == "__main__":
    if False:
        ta3n_training_script.main()
    else:
        # 3: Start the sweep
        sweep_id = wandb.sweep(
            sweep=dataset_shift_configuration,
            project='mldl23-ego-ta3n',
        )
        wandb.agent(sweep_id, function=ta3n_training_script.main)
