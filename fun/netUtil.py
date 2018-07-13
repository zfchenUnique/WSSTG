import os
import torch
import shutil
def save_check_point(state, is_best=False, file_name='../data/models/checkpoint.pth'):
    torch.save(state, file_name)
    if is_best:
        shutil.copyfile(file_name, '../data/model/best_model.pth')

def load_model_state(model, file_name):
    states = torch.load(file_name)
    model.load_state_dict(states)


