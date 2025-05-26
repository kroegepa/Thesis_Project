import torch
def pof_transform(batch, task_name):
    #Add supported tasks here
    if task_name not in (
        "pursuit",
    ):
        raise NotImplementedError(f"Task {task_name} not supported for POF transform.")
    
    #Should make this part of the task object!

    
    batch = pursuit_grouping(batch)
    return batch
def calc_agent_position(observation_size):
    #I think this is only right for uneven observation sizes
    return (observation_size // 2) + 1
def pursuit_grouping(batch):
    # Placeholder: determine which agents are near an evader based on obs and action
    #Batch Time and agent dimension
    obs = batch["pursuer"]["observation"]  # shape: [B, T, A, obs_dim]
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
                if abs(reward[b,t,a,0]) > 2:  # Threshold for grouping
                    grouping_tensor[b,t,a,2] = 1
                elif abs(reward[b,t,a,0]) >= -0.9:  # Threshold for grouping
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
    reward = batch["next"]["pursuer"]["reward"]  # shape: [B, T, A, 1]
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
    batch["next"]["pursuer"]["reward"] = new_reward
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