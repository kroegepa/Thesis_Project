#Naming???
import torch
def pof_transform(batch, task_name):
    #Add supported tasks here
    if task_name not in (
        "pursuit",
    ):
        raise NotImplementedError(f"Task {task_name} not supported for POF transform.")
    
    #Will make this part of the task object!

    
    batch = pursuit_grouping(batch)
    return batch

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
    for b in range(reward.shape[0]):
        for t in range(reward.shape[1]-1):
            for a in range(reward.shape[2]):
                #Eww
                if abs(reward[b,t,a,0]) > 3:
                    grouping_tensor[b,t,a,2] = 1
                elif touching_distance(obs[b,t+1,a,:]):
                    grouping_tensor[b,t,a,1] = 1
                else:
                    grouping_tensor[b,t,a,0] = 1

   # reward: [B, T, A, 1]
    new_reward = reward.clone()

    # Loop over groups 0, 1, 2
    for g in range(3):
        # Create mask for group g: shape [B, T, A, 1]
        group_mask = (grouping_tensor[..., g] == 1).unsqueeze(-1).float()

        # Compute sum and count for averaging
        group_sum = (reward * group_mask).sum(dim=2, keepdim=True)  # sum over agents
        group_count = group_mask.sum(dim=2, keepdim=True).clamp(min=1)  # avoid divide-by-zero

        group_mean = group_sum / group_count  # shape: [B, T, 1, 1]

        # Assign average back to each agent in group
        new_reward[group_mask.bool()] = group_mean.expand_as(new_reward)[group_mask.bool()]
    
    # Save new reward back into batch
    batch["next"]["pursuer"]["reward"] = new_reward
    return batch

def touching_distance(observation):
    #Hardcoded for obs size = 7 for now
    if (observation[3,4,2] == 1 or 
    observation[5,4,2] == 1 or 
    observation[4,3,2] == 1 or
    observation[4,5,2] == 1):
        return True
    else:
        return False    
