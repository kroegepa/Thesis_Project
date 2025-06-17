import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture



def pof_transform(batch, task_name):
    #Add supported tasks here
    if task_name not in (
        "pursuit",
    ):
        raise NotImplementedError(f"Task {task_name} not supported for POF transform.")
    
    #Should make this part of the task object!

    if task_name == "pursuit":
        batch = pursuit_grouping(batch)
    if task_name == "simeple_spread":
        batch = spread_grouping(batch)
    return batch


class SpreadGroupingMLP(nn.Module):
    def __init__(self, obs_dim=30, act_dim=5, hidden_dim=128, num_groups=3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_groups)
        )

    def forward(self, obs, action):
        """
        obs: [N, obs_dim]
        action: [N, act_dim]
        returns: [N, num_groups] logits
        """
        x = torch.cat([obs, action], dim=-1)  # [N, obs_dim + act_dim]
        return self.fc(x)  # [N, num_groups]

class PursuitGroupingCNN(nn.Module):
    def __init__(self, act_dim=5, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 7x7x3 → 7x7x16
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 7x7x32
            nn.ReLU(),
            nn.Flatten()  # 7*7*32 = 1568
        )
        self.fc = nn.Sequential(
            nn.Linear(1568 + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, obs_grid, action):
        """
        obs_grid: [N, 3, 7, 7]  (C, H, W)
        action:   [N, act_dim]
        """
        x = self.conv(obs_grid)
        x = torch.cat([x, action], dim=-1)
        return self.fc(x)  # [N, num_classes]

# class GroupingNet(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden=128, out_dim=3):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim + act_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, out_dim)
#         )

#     def forward(self, obs, act):
#         x = torch.cat([obs, act], dim=-1)
#         return self.net(x)  # shape: [N, 3]

def calc_agent_position(observation_size):
    #I think this is only right for uneven observation sizes
    return (observation_size // 2) + 1

def oracle_grouping(batch, task_name):
    if task_name == "pursuit":
        return pursuit_grouping(batch)
    elif task_name == "simple_spread":
        return spread_grouping(batch)
    else:
        raise NotImplementedError(f"Task {task_name} not supported for oracle grouping.")

def spread_grouping(batch):
    reward = batch["next"]["agent"]["reward"]  # shape: [B, T, A, 1]
    grouping_tensor = torch.zeros((*reward.shape[:-1], 3), dtype=torch.float32)

    for b in range(reward.shape[0]):
        for t in range(reward.shape[1]):
            rewards_t = reward[b, t, :, 0]  # shape: [A]
            unique_vals = rewards_t.unique().sort().values  # up to 3 unique rewards

            for g, val in enumerate(unique_vals[:3]):  # up to 3 groups
                mask = rewards_t == val
                grouping_tensor[b, t, mask, g] = 1

    return grouping_tensor  # shape: [B, T, A, 3]


def pursuit_grouping_fuzzy(batch, n_groups=3):
    """
    Generate grouping tensor using a Gaussian Mixture Model fit on the noisy reward signal.
    Produces one-hot hard assignments (argmax over soft probabilities).
    
    Args:
        batch (TensorDict): contains ["next"]["pursuer"]["reward"] of shape [B, T, A, 1]
        n_groups (int): number of clusters to create (default 3)
        
    Returns:
        grouping_tensor: [B, T, A, n_groups] with one-hot group assignments
    """
    reward = batch["next"]["pursuer"]["reward"]  # shape: [B, T, A, 1]
    B, T, A, _ = reward.shape

    # Flatten to [N, 1]
    rewards_np = reward.view(-1, 1).detach().cpu().numpy()

    # Fit GMM on all rewards in batch
    gmm = GaussianMixture(n_components=n_groups, covariance_type="full", random_state=0)
    gmm.fit(rewards_np)

    # Predict group (hard assignment)
    group_ids = gmm.predict(rewards_np)  # shape: [N]

    # Convert to one-hot
    group_tensor = torch.nn.functional.one_hot(torch.tensor(group_ids), num_classes=n_groups).float()  # [N, n_groups]
    grouping_tensor = group_tensor.view(B, T, A, n_groups).to(reward.device)

    return grouping_tensor


def pursuit_grouping(batch):
    # Placeholder: determine which agents are near an evader based on obs and action
    #Batch Time and agent dimension
    #obs = batch["pursuer"]["observation"]  # shape: [B, T, A, obs_dim]
    #action = batch["pursuer"]["action"]  # shape: [B, T, A, action_dim]
    reward = batch["next"]["pursuer"]["reward"]  # shape: [B, T, A, 1]
    grouping_tensor = torch.zeros((*reward.shape[:-1],3),dtype=torch.float32)
    #Just time dimension
    #max_group_vector = torch.zeros(reward.shape[1])
    #Use batch size from batch object here instead?

    #Perfect grouping by reward signal
    for b in range(reward.shape[0]):
        for t in range(reward.shape[1]):
            for a in range(reward.shape[2]):
                if reward[b,t,a,0] > 2:  # Threshold for grouping
                    grouping_tensor[b,t,a,2] = 1
                elif reward[b,t,a,0] > -0.1:  # Threshold for grouping
                    grouping_tensor[b,t,a,1] = 1
                else:
                    grouping_tensor[b,t,a,0] = 1
    return grouping_tensor
    #Simple Heuristic
    agent_position = calc_agent_position(obs.shape[3])
    for b in range(reward.shape[0]):
        for t in range(reward.shape[1]):
            for a in range(reward.shape[2]):
                if touching_distance(obs[b,t,a,:],agent_position,agent_position):
                    grouping_tensor[b,t,a,1] = 1
                else:
                    grouping_tensor[b,t,a,0] = 1

    return grouping_tensor


def grouping_reward_averaging(batch, grouping_tensor):
    reward = batch["next"]["pursuit"]["reward"]  # shape: [B, T, A, 1]
    # Loop over groups 0, 1, 2
    new_reward = reward.clone()
    device = reward.device
    for g in range(grouping_tensor.shape[-1]):
        # Create mask for group g: shape [B, T, A, 1]
        group_mask = (grouping_tensor[..., g] == 1).unsqueeze(-1).float().to(device)  # shape: [B, T, A, 1]

        # Compute sum and count for averaging
        group_sum = (reward * group_mask).sum(dim=2, keepdim=True)  # sum over agents
        group_count = group_mask.sum(dim=2, keepdim=True).clamp(min=1)  # avoid divide-by-zero

        group_mean = group_sum / group_count  # shape: [B, T, 1, 1]

        # Assign average back to each agent in group
        new_reward[group_mask.bool()] = group_mean.expand_as(new_reward)[group_mask.bool()]
    
    # Save new reward back into batch
    batch["next"]["pursuit"]["reward"] = new_reward
    return batch

def touching_distance(observation,x,y):
    #Hardcoded for obs size = 7 for now
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0,0)]:
        nx, ny = x + dx, y + dy
        #TODO Size constraint observation
        if observation[nx, ny, 2] == 1:  # Check evader channel
            return True
    return False    

#Doesnt matter
def catching(observation, x, y):
    #Under Construction
    return False
    #sth like this
    surrounding_count = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < observation.shape[0] and 0 <= ny < observation.shape[1]:
            if observation[nx, ny, 1] == 1:  # Check pursuer channel (fixed)
                surrounding_count += 1

    return surrounding_count >= 3