import os
from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore

model = AutoModelForScore.from_pretrained('PKU-Alignment/beaver-7b-v1.0-reward', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('PKU-Alignment/beaver-7b-v1.0-reward', use_fast=False)

input = 'BEGINNING OF CONVERSATION: USER: hello ASSISTANT:Hello! How can I help you today?'

input_ids = tokenizer(input, return_tensors='pt')
output = model(**input_ids)
print(output)

# ScoreModelOutput(
#     scores=tensor([[[-19.6476],
#         [-20.2238],
#         [-21.4228],
#         [-19.2506],
#         [-20.2728],
#         [-23.8799],
#         [-22.6898],
#         [-21.5825],
#         [-21.0855],
#         [-20.2068],
#         [-23.8296],
#         [-21.4940],
#         [-21.9484],
#         [-13.1220],
#         [ -6.4499],
#         [ -8.1982],
#         [ -7.2492],
#         [ -9.3377],
#         [-13.5010],
#         [-10.4932],
#         [ -9.7837],
#         [ -6.4540],
#         [ -6.0084],
#         [ -5.8093],
#         [ -6.6134],
#         [ -5.8995],
#         [ -9.1505],
#         [-11.3254]]], grad_fn=<ToCopyBackward0>),
#     end_scores=tensor([[-11.3254]], grad_fn=<ToCopyBackward0>)
# )
import torch
import tensorflow as tf
import onnx
import discord
from discord.ext import commands
import gym

env = gym.make('Taxi-v3')

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000

# Initialize Q-table with zeros
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Q-learning algorithm
for episode in range(num_episodes):
    state = env.reset()
    done = False

