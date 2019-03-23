import random
import torch
import numpy as np

# This function takes a batch of input frame stacks of dtype=uint8, and returns
# that batch converted to float / 255
def prep_input(x, noise=False, epsilon=0.05, chance=0.5, device='cpu'):
    x = x.float() / 255
    if noise:
        rand_sample = random.uniform(0, 1)
        if rand_sample <= chance:
            rand_epsilon = random.uniform(0, epsilon)
            x = uniform(x, rand_epsilon, device)
    return x


# This function applies uniform noise to an input.
def uniform(input, epsilon, device):
    noise = torch.tensor(np.random.uniform(-epsilon, epsilon, input.size()), dtype=torch.float32, device=device)
    # sign_noise = noise.sign()
    # perturbed_input = input + epsilon * sign_noise
    perturbed_input = input + noise
    perturbed_input = torch.clamp(perturbed_input, 0, 1)
    return perturbed_input
