


def get_mean_std(loader):
    channel_sum, channel_sq_sum, num_batches = 0,0,0

    for data, _ in loader:
        channel_sum += torch.mean(data, dim[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim[0,2,3])
        num_batches += 1
    
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches - mean**2)**0.5

    return mean, std
