import numpy as np
import random
from collections import namedtuple
import torch
import torch.nn as nn
from net2brain.utils.download_datasets import DatasetNSD_872
from net2brain.feature_extraction import FeatureExtractor
from net2brain.evaluations.ridge_regression import Ridge_Encoding
import os
import time
import shutil
import wandb

State = namedtuple('State', ['layer_type', 'layer_depth', 'num_filters', 'kernel_size', 'fc_count'])
Action = namedtuple('Action', ['layer_type', 'num_filters', 'kernel_size', 'skip_connection'])

def delete_folders(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
                print(f"Deleted folder: {item_path}")
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")

class CNNArchitectureSampler:
    def __init__(self, max_depth=20, input_shape=(3, 224, 224), roi_path=None, stimuli_path=None):
        self.max_depth = max_depth
        self.input_shape = input_shape
        self.layer_types = ['conv', 'pool', 'fc', 'output']
        self.num_filters_options = [16, 32, 64, 128, 256, 512, 768] # maybe add a constraint
        self.kernel_sizes = [1, 3, 5, 7]
        self.skip_connection_options = [True, False]
        self.roi_path = roi_path
        self.stimuli_path = stimuli_path

        if input_shape[0] not in self.num_filters_options:
            self.num_filters_options = [input_shape[0]] + self.num_filters_options

        self.state_space = len(self.layer_types) * max_depth * len(self.num_filters_options) * len(self.kernel_sizes) * 3
        self.action_space = (len(self.layer_types) * 
                             len(self.num_filters_options) * 
                             len(self.kernel_sizes) * 
                             len(self.skip_connection_options))

        self.q_table = np.zeros((self.state_space, self.action_space))
        self.epsilon = 1.0
        self.alpha = 0.1
        self.gamma = 1.0
        self.total_models_sampled = 0
        self.sampled_architectures = {}
        self.episode = 0
        self.total_architectures_sampled = 0
        self.run = wandb.init(project="shallow-brain-dp-nas", config={
                        "max_depth": 20,
                        "input_shape": (3, 224, 224),
                   })
        self.best_architecture = None
        self.best_reward = float('-inf')

    def update_epsilon(self, archs_per_episode):
        self.epsilon = max(0.1, 1.0 - (self.episode * archs_per_episode) // archs_per_episode * 0.1)

    def state_to_index(self, state):
        layer_type_idx = self.layer_types.index(state.layer_type) if state.layer_type != -1 else 0
        num_filters_idx = self.num_filters_options.index(state.num_filters) if state.num_filters in self.num_filters_options else 0
        kernel_size_idx = self.kernel_sizes.index(state.kernel_size) if state.kernel_size in self.kernel_sizes else 0
        return (layer_type_idx * self.max_depth * len(self.num_filters_options) * len(self.kernel_sizes) * 3 +
                state.layer_depth * len(self.num_filters_options) * len(self.kernel_sizes) * 3 +
                num_filters_idx * len(self.kernel_sizes) * 3 +
                kernel_size_idx * 3 +
                state.fc_count)

    def action_to_index(self, action):
        layer_type_idx = self.layer_types.index(action.layer_type)
        num_filters_idx = self.num_filters_options.index(action.num_filters) if action.layer_type != 'pool' else 0
        kernel_size_idx = self.kernel_sizes.index(action.kernel_size) if action.layer_type in ['conv', 'pool'] else 0
        skip_idx = int(action.skip_connection)
        return (layer_type_idx * len(self.num_filters_options) * len(self.kernel_sizes) * 2 +
                num_filters_idx * len(self.kernel_sizes) * 2 +
                kernel_size_idx * 2 +
                skip_idx)

    def index_to_action(self, index):
        skip_idx = index % 2
        index //= 2
        kernel_idx = index % len(self.kernel_sizes)
        index //= len(self.kernel_sizes)
        num_filters_idx = index % len(self.num_filters_options)
        index //= len(self.num_filters_options)
        layer_type_idx = index

        layer_type = self.layer_types[layer_type_idx]
        num_filters = self.num_filters_options[num_filters_idx] if layer_type != 'pool' else 0
        kernel_size = self.kernel_sizes[kernel_idx] if layer_type in ['conv', 'pool'] else 0

        return Action(
            layer_type=layer_type,
            num_filters=num_filters,
            kernel_size=kernel_size,
            skip_connection=bool(skip_idx)
        )

    def sample_action(self, state):
        state_idx = self.state_to_index(state)
        if random.random() < self.epsilon:
            return self.index_to_action(random.randint(0, self.action_space - 1))
        else:
            action_idx = np.argmax(self.q_table[state_idx])
            return self.index_to_action(action_idx)

    def is_valid_action(self, state, action):
        if action.layer_type == 'output':
            return state.layer_depth > 0  # Only allow output if we have at least one layer
        if state.layer_depth >= self.max_depth - 1:
            return action.layer_type == 'output'
        if state.layer_type == 'pool' and action.layer_type == 'pool':
            return False
        if state.layer_type == 'fc' and action.layer_type in ['conv', 'pool']:
            return False
        if action.skip_connection and (state.layer_type == 'fc' or action.layer_type == 'fc'):
            return False
        return True
    
    def architecture_to_torch_model(self, architecture, skip_connections):
        class CNNModel(nn.Module):
            def __init__(self, arch, skips, input_shape):
                super(CNNModel, self).__init__()
                self.layers = nn.ModuleList()
                self.skips = []
                
                in_channels = input_shape[0]
                current_size = input_shape[1]
                fc_input_size = 0
                is_flattened = False
                
                for i, (layer_type, num_filters, kernel_size) in enumerate(arch):
                    if layer_type == 'conv':
                        if is_flattened:
                            break  # Stop adding layers if we've already flattened
                        self.layers.append(nn.Conv2d(in_channels, num_filters, kernel_size, padding=kernel_size//2))
                        self.layers.append(nn.LeakyReLU())
                        in_channels = num_filters
                    elif layer_type == 'pool':
                        if is_flattened:
                            break  # Stop adding layers if we've already flattened
                        self.layers.append(nn.MaxPool2d(2))
                        current_size //= 2
                    elif layer_type == 'fc':
                        if not is_flattened:
                            fc_input_size = in_channels * current_size * current_size
                            self.layers.append(nn.Flatten())
                            is_flattened = True
                        self.layers.append(nn.Linear(fc_input_size, num_filters))
                        self.layers.append(nn.LeakyReLU())
                        fc_input_size = num_filters
                
                # Remove the last ReLU layer to allow for proper output scaling
                if isinstance(self.layers[-1], nn.LeakyReLU):
                    self.layers = self.layers[:-1]
                
                # Filter valid skip connections
                for start, end in skips:
                    if start < end and start < len(self.layers) and end < len(self.layers):
                        self.skips.append((start, end))
            
            def forward(self, x):
                skip_outputs = {}
                for i, layer in enumerate(self.layers):
                    if i in [s[0] for s in self.skips]:
                        skip_outputs[i] = x
                    if i in [s[1] for s in self.skips]:
                        skip_source = [s[0] for s in self.skips if s[1] == i]
                        if skip_source and skip_source[0] in skip_outputs:
                            skip_x = skip_outputs[skip_source[0]]
                            if x.shape != skip_x.shape:
                                print(f"Shape mismatch at layer {i}: x: {x.shape}, skip: {skip_x.shape}")
                                # Skip this connection if shapes don't match
                                continue
                            x = x + skip_x
                    x = layer(x)
                return x

        return CNNModel(architecture, skip_connections, self.input_shape)

    def sample_architecture(self):
        architecture = []
        skip_connections = []
        state = State(layer_type='conv', layer_depth=0, num_filters=self.input_shape[0], kernel_size=3, fc_count=0)

        # Force first layer to be convolutional
        action = Action(layer_type='conv', num_filters=random.choice(self.num_filters_options), 
                        kernel_size=random.choice(self.kernel_sizes), skip_connection=False)
        architecture.append((action.layer_type, action.num_filters, action.kernel_size))
        state = State(layer_type=action.layer_type, layer_depth=1, 
                    num_filters=action.num_filters, kernel_size=action.kernel_size, fc_count=0)

        while True:
            valid_action = False
            while not valid_action:
                action = self.sample_action(state)
                valid_action = self.is_valid_action(state, action)

            if action.layer_type == 'output' or state.layer_depth >= self.max_depth - 1:
                break

            if action.layer_type == 'conv':
                # Ensure the number of filters is valid
                action = action._replace(num_filters=max(action.num_filters, state.num_filters))

            architecture.append((action.layer_type, action.num_filters, action.kernel_size))
            
            # Sample skip connections
            if len(architecture) > 1 and action.skip_connection:
                possible_skip_starts = [i for i, (layer_type, _, _) in enumerate(architecture[:-1])
                                        if layer_type in ['conv', 'pool']]
                if possible_skip_starts:
                    skip_start = random.choice(possible_skip_starts)
                    skip_connections.append((skip_start, len(architecture) - 1))
            
            new_fc_count = state.fc_count + 1 if action.layer_type == 'fc' else state.fc_count
            state = State(
                layer_type=action.layer_type,
                layer_depth=state.layer_depth + 1,
                num_filters=action.num_filters,
                kernel_size=action.kernel_size,
                fc_count=new_fc_count
            )

        return architecture, skip_connections

    def architecture_to_torch_model(self, architecture, skip_connections):
        class CNNModel(nn.Module):
            def __init__(self, arch, skips, input_shape):
                super(CNNModel, self).__init__()
                self.layers = nn.ModuleList()
                self.skips = []
                
                in_channels = input_shape[0]
                current_size = input_shape[1]
                fc_input_size = 0
                is_flattened = False
                
                if not arch:  # If the architecture is empty, add a default layer
                    self.layers.append(nn.Conv2d(in_channels, 64, 3, padding=1))
                    self.layers.append(nn.ReLU())
                    self.layers.append(nn.Flatten())
                    fc_input_size = 64 * current_size * current_size
                else:
                    for i, (layer_type, num_filters, kernel_size) in enumerate(arch):
                        if layer_type == 'conv':
                            if is_flattened:
                                break  # Stop adding layers if we've already flattened
                            self.layers.append(nn.Conv2d(in_channels, num_filters, kernel_size, padding=kernel_size//2))
                            self.layers.append(nn.ReLU())
                            in_channels = num_filters
                        elif layer_type == 'pool':
                            if is_flattened:
                                break  # Stop adding layers if we've already flattened
                            self.layers.append(nn.MaxPool2d(2))
                            current_size //= 2
                        elif layer_type == 'fc':
                            if not is_flattened:
                                fc_input_size = in_channels * current_size * current_size
                                self.layers.append(nn.Flatten())
                                is_flattened = True
                            self.layers.append(nn.Linear(fc_input_size, num_filters))
                            self.layers.append(nn.ReLU())
                            fc_input_size = num_filters
                            in_channels = num_filters  # Update in_channels for potential future layers
                    
                    # If we haven't added any FC layers, add a Flatten layer, followed by a default FC layer
                    #if not is_flattened:
                    #    self.layers.append(nn.Flatten())
                    #    fc_input_size = in_channels * current_size * current_size
                    #    self.layers.append(nn.Linear(fc_input_size, 4096))

                
                # Filter valid skip connections
                for start, end in skips:
                    if start < end and start < len(self.layers) and end < len(self.layers):
                        self.skips.append((start, end))
            
            def forward(self, x):
                skip_outputs = {}
                for i, layer in enumerate(self.layers):
                    if i in [s[0] for s in self.skips]:
                        skip_outputs[i] = x
                    if i in [s[1] for s in self.skips]:
                        skip_source = [s[0] for s in self.skips if s[1] == i]
                        if skip_source and skip_source[0] in skip_outputs:
                            skip_x = skip_outputs[skip_source[0]]
                            if x.shape != skip_x.shape:
                                # Adjust the skip connection to match the current tensor shape
                                skip_x = self.adjust_skip_connection(skip_x, x.shape)
                            x = x + skip_x
                    x = layer(x)
                return x
            
            # AlexNet-like weight initialization
            def init_weights(self):
                for layer in self.layers:
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
                    elif isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                        nn.init.constant_(layer.bias, 0)

            def adjust_skip_connection(self, skip_x, target_shape):
                # Adjust the number of channels and spatial dimensions if needed
                if skip_x.shape[1] != target_shape[1]:
                    # Adjust number of channels
                    skip_x = nn.functional.conv2d(skip_x, torch.randn(target_shape[1], skip_x.shape[1], 1, 1))
                if skip_x.shape[2:] != target_shape[2:]:
                    # Adjust spatial dimensions
                    skip_x = nn.functional.interpolate(skip_x, size=target_shape[2:])
                return skip_x

        return CNNModel(architecture, skip_connections, self.input_shape)

    def evaluate_model(self, model):
        fx = FeatureExtractor(model=model, device='mps')
        
        # Generate a unique identifier
        unique_id = f"{os.getpid()}_{int(time.time() * 1000)}"
        save_path = f"tmp/temp_res_{unique_id}"
        
        # get last named layer
        last_layer_name = list(model.layers._modules.keys())[-1]
      
        last_layer = [f"layers.{last_layer_name}"]
        print(f"Last layer name: {last_layer}")

        fx.extract(data_path=self.stimuli_path, save_path=save_path, consolidate_per_layer=False, layers_to_extract=last_layer)
        

        # TODO random state for reproducibility to 1,2,3
        results_dataframe = Ridge_Encoding(
            save_path,
            self.roi_path,
            f"temp_model_{unique_id}",
            n_folds=3,
            trn_tst_split=0.8,
            n_components=100,
            batch_size=100,
            return_correlations=True,
            save_path=f"tmp/results_{unique_id}",
            alpha=1.0,
        )
        
        last_layer_df = results_dataframe[results_dataframe.Layer == last_layer[0]]
    
        mean = last_layer_df.R.mean()
        print(f"Mean R value: {mean}")

        # remove the folder
        shutil.rmtree(save_path)
        shutil.rmtree(f"tmp/results_{unique_id}")

        return mean

    def update_q_value(self, state, action, next_state, reward):
        state_idx = self.state_to_index(state)
        action_idx = self.action_to_index(action)
        next_state_idx = self.state_to_index(next_state)

        current_q = self.q_table[state_idx, action_idx]
        next_max_q = np.max(self.q_table[next_state_idx])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_idx, action_idx] = new_q

    def sample_and_evaluate(self):
        architecture, skip_connections = self.sample_architecture()
        print(architecture)

        arch_key = (tuple(architecture), tuple(skip_connections))
        
        if arch_key in self.sampled_architectures:
            reward = self.sampled_architectures[arch_key]
            return architecture, skip_connections, reward, True
        
        model = self.architecture_to_torch_model(architecture, skip_connections)
        print(model)

        rewards = []
        for _ in range(3):
            model.init_weights()
            reward = self.evaluate_model(model)
            rewards.append(reward)

        # remove NaN values from the list
        rewards = [r for r in rewards if not np.isnan(r)]

        if len(rewards) == 0:
            reward = 0
        else:
            reward = np.mean(rewards)
        print(reward)
        
        self.sampled_architectures[arch_key] = reward
        
        return architecture, skip_connections, reward, False

    def train(self, num_episodes=10, models_per_episode=100):
        for episode in range(num_episodes):
            self.episode = episode
            
            architectures = []
            skip_connections_list = []
            rewards = []
            new_samples = 0

            for _ in range(models_per_episode):
                arch, skips, reward, is_cached = self.sample_and_evaluate()
                architectures.append(arch)
                skip_connections_list.append(skips)
                rewards.append(reward)
                if not is_cached:
                    new_samples += 1
                    self.total_architectures_sampled += 1

            self.run.log({"rewards": rewards})
            
            for architecture, skip_connections, reward in zip(architectures, skip_connections_list, rewards):
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_architecture = (architecture, skip_connections)

                for i in range(len(architecture) - 1):
                    state = State(layer_type=architecture[i][0], layer_depth=i, 
                                num_filters=architecture[i][1], kernel_size=architecture[i][2],
                                fc_count=sum(1 for layer in architecture[:i+1] if layer[0] == 'fc'))
                    next_state = State(layer_type=architecture[i+1][0], layer_depth=i+1,
                                    num_filters=architecture[i+1][1], kernel_size=architecture[i+1][2],
                                    fc_count=sum(1 for layer in architecture[:i+2] if layer[0] == 'fc'))
                    action = Action(
                        layer_type=architecture[i][0],
                        num_filters=architecture[i][1],
                        kernel_size=architecture[i][2],
                        skip_connection=any(s[0] == i or s[1] == i for s in skip_connections)
                    )
                    
                    self.update_q_value(state, action, next_state, reward)

            self.update_epsilon(models_per_episode)
            print(f"Epsilon now is: {self.epsilon:.2f}")

            self.run.log({
                "episode": episode,
                "best_reward": max(rewards),
                "mean_reward": sum(rewards) / len(rewards),
                "epsilon": self.epsilon,
                "new_architectures_sampled": new_samples,
                "total_unique_architectures": len(self.sampled_architectures),
            })

            print(f"Episode {episode + 1} completed. Best reward: {max(rewards):.4f}, Epsilon: {self.epsilon:.2f}")
            print(f"New architectures sampled this episode: {new_samples}")
            print(f"Total unique architectures sampled: {len(self.sampled_architectures)}")

        # Print the best architecture at the end of training
        print("\nBest architecture found:")
        architecture, skip_connections = self.best_architecture
        for i, (layer_type, num_filters, kernel_size) in enumerate(architecture):
            print(f"  Layer {i}: Type={layer_type}, Filters={num_filters}, Kernel Size={kernel_size}")
        print("Skip connections:", skip_connections)
        print(f"Best reward: {self.best_reward:.4f}")

        # Create and print the best model
        best_model = self.architecture_to_torch_model(architecture, skip_connections)
        print("\nBest Model Summary:")
        print(best_model)

        trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
        print(f"\nTrainable parameters: {trainable_params:,}")

        return best_model

if __name__ == "__main__":
    paths_NSD_872 = DatasetNSD_872().load_dataset()

    stimuli_path = paths_NSD_872["NSD_872_images"]
    roi_path = paths_NSD_872["NSD_872_fmri"] 

    roi_path = os.path.join(roi_path, "prf-visualrois/V2combined")

    sampler = CNNArchitectureSampler(max_depth=20, input_shape=(3, 224, 224), roi_path=roi_path, stimuli_path=stimuli_path)
    
    best_model = sampler.train(num_episodes=10, models_per_episode=150)

    print("Training complete. Best model...")
    print(best_model)