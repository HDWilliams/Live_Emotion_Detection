#%%
import torch
print(torch)
# %%
torch.tensor([4])
# %%
import torch
from emotion_cnn import CNN

model = CNN()
model.load_state_dict(torch.load('model0.6804249390456287.pth'))
print(model)
# %%
