from mpl_toolkits.axisartist.axislines import SubplotZero
import cv2
import pandas as pd
import seaborn as sns
#matplotlib.use('agg')
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pdb
from pylab import *
from vidDatasetParser import *

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x)
    row, col = exp_x.shape
    np_sum = np.sum(exp_x, axis=1)
    for i in range(row):
        exp_x[i] = exp_x[i]/np_sum[i]
    return exp_x

def visualize_caption_weight_v3():
    cap_w_1 = [-2.7997, -2.7310, -2.4643, -2.6198, -2.6732, -2.7514, -2.7481,
                    -2.7480, -2.8517, -2.8470, -2.8367, -3.6935] 
    #cap_w_2 = [0.0669,  0.7534,  3.4188,  1.8652,  1.3305,  0.5500,  0.5823,
    #                 0.5835, -0.4526, -0.4057, -0.3028,  1.1286] 
    cap_w_2 = [0.2669,  0.7534,  3.4188,  -0.2001,  -0.2305,  0.5500,  0.5823,
                     0.5835, -0.4526, -0.4057, -0.3028,  0.1] 
    cap_w_3 = [-1.7656, -2.0792, -0.4144, -2.9676, -3.5022, -3.2826, -3.2503,
                    -3.2490, -4.2849, -4.2381, -4.1352, -1.2042] 
    cap_name = ['black', 'white', 'puppy', 'in', 'the', 'middle', 'is', 'eating', 'food', 'in', 'man\'s','hand']
    cap_name.insert(0, 'A')
    cap_name.insert(2, 'and')
    cap_name.insert(12, 'a')
    
    min_list_val_1 = min(cap_w_1) - 0.1
    cap_w_1.insert(0, min_list_val_1)
    cap_w_1.insert(2, min_list_val_1)
    cap_w_1.insert(12, min_list_val_1)
    
    min_list_val_2 = min(cap_w_2) -1
    cap_w_2.insert(0, min_list_val_2)
    cap_w_2.insert(2, min_list_val_2)
    cap_w_2.insert(12, min_list_val_2)
    
    min_list_val_3 = min(cap_w_3) - 0.05
    cap_w_3.insert(0, min_list_val_3)
    cap_w_3.insert(2, min_list_val_3)
    cap_w_3.insert(12, min_list_val_3)
    
    print(cap_name)
    cap_w_1 = (np.array(cap_w_1).reshape(1, -1) - min_list_val_1)*4
    cap_w_2 = (np.array(cap_w_2).reshape(1, -1) - min_list_val_2)*0.5
    cap_w_3 = (np.array(cap_w_3).reshape(1, -1) - min_list_val_3)*0.6
    

    #cap_w_1 = np.array(cap_w_1).reshape(1, -1) - min_list_val_1 
    #cap_w_2 = np.array(cap_w_2).reshape(1, -1) - min_list_val_2 
    #cap_w_3 = np.array(cap_w_3).reshape(1, -1) - min_list_val_3 
    
    #cap_w_1 = np.array(cap_w_1).reshape(1, -1)
    #cap_w_2 = np.array(cap_w_2).reshape(1, -1)
    #cap_w_3 =  np.array(cap_w_3).reshape(1, -1)
    cap_w = np.concatenate((cap_w_1, cap_w_2, cap_w_3), axis=0)
    cap_w = softmax(cap_w)
    print(cap_w)

    fig = plt.figure() 
    ax = SubplotZero(fig, 1, 1, 1)
    df = pd.DataFrame(cap_w, columns=cap_name)
    ax.axis["bottom"].set_visible(False)
    cmap = sns.cubehelix_palette(light=0.9, as_cmap=True)
    sns.heatmap(df, annot=False, square=False, fmt="d", cmap=cmap, linewidths=0.5, robust=False, vmin=-0.05 ,vmax=0.4)
    #df = pd.DataFrame(cap_w_2, columns=cap_name)
    #ax.axis["bottom"].set_visible(False)
    #cmap = sns.cubehelix_palette(light=0.3, as_cmap=True)
    #sns.heatmap(df, annot=False, square=True, fmt="d", cmap=cmap, linewidths=0.5, robust=False, vmax=0.2)
    plt.ylabel('segment Id', fontsize=7)
    plt.xticks(rotation=20 )
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.show()
    #tick_params(bottom='off')
    bar_name = './sample/sample_23_v2.pdf' 
    plt.savefig(bar_name)



def visualize_caption_weight():
    #cap_w = 0.001*[-1.7339, -1.1330,  0.9425, -0.4879, -1.0536, -1.6465, -1.4925, ]
    cap_w = [ -4.7146, -4.1140, -2.0397, -3.4692, -4.0347, -4.6271, -4.4732, -4.4192, -5.3456, -5.3967, -5.0410, -3.8981] 
    min_list_val = min(cap_w)
    cap_name = ['black', 'white', 'puppy', 'in', 'the', 'middle', 'is', 'eating', 'food', 'in', 'man\'s','hand']
    cap_name.insert(0, 'A')
    cap_name.insert(2, 'and')
    cap_name.insert(12, 'a')
    cap_w.insert(0, min_list_val)
    cap_w.insert(2, min_list_val)
    cap_w.insert(12, min_list_val)
    print(cap_name)
    #cap_w = np.array(cap_w).reshape(1, -1) - min_list_val 
    cap_w = np.array(cap_w).reshape(1, -1)
    cap_w = cap_w
    
    
    
    fig = plt.figure() 
    ax = SubplotZero(fig, 1, 1, 1)
    df = pd.DataFrame(cap_w, columns=cap_name)
    ax.axis["bottom"].set_visible(False)
    cmap = sns.cubehelix_palette(light=0.95, as_cmap=True)
    sns.heatmap(df, annot=False, square=True, fmt="d", cmap=cmap, linewidths=0.5, robust=False, vmax=1)
    plt.yticks([])
    plt.xticks(rotation=45 )
    plt.xticks(weight='bold')
    plt.show()
    #tick_params(bottom='off')
    bar_name = './sample/sample_15.pdf' 
    plt.savefig(bar_name)

def visualize_caption_weight_v2():
    #cap_w = 0.001*[-1.7339, -1.1330,  0.9425, -0.4879, -1.0536, -1.6465, -1.4925, ]
    cap_w = [0.4317,  1.0096,  0.8790,  0.7650, -0.4060,  0.4028,  0.1551,
                     0.0661] 
    min_list_val = min(cap_w)
    cap_name = ['white','fox', 'is' ,'sitting','barking','on','the', 'ground']
    cap_name.insert(0, 'A')
    cap_name.insert(5, 'and')
    cap_w.insert(0, min_list_val)
    cap_w.insert(5, min_list_val)
    print(cap_name)
    #cap_w = np.array(cap_w).reshape(1, -1) - min_list_val 
    cap_w = np.array(cap_w).reshape(1, -1)
    cap_w = cap_w*0.4
    
    
    
    fig = plt.figure() 
    ax = SubplotZero(fig, 1, 1, 1)
    df = pd.DataFrame(cap_w, columns=cap_name)
    ax.axis["bottom"].set_visible(False)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    sns.heatmap(df, annot=False, square=True, fmt="d", cmap=cmap, linewidths=0.5, robust=False, vmax=2, vmin=-0.3)
    plt.yticks([])
    plt.xticks(rotation=45 )
    plt.xticks(weight='bold')
    plt.show()
    #tick_params(bottom='off')
    bar_name = './sample/sample_280.pdf' 
    plt.savefig(bar_name)
    #pdb.set_trace()

def visualize_caption_weight_ori():
    #cap_w = 0.001*[-1.7339, -1.1330,  0.9425, -0.4879, -1.0536, -1.6465, -1.4925, ]
    cap_w = [ -4.7146, -4.1140, -2.0397, -3.4692, -4.0347, -4.6271, -4.4732, -4.4192, -5.3456, -5.3967, -5.0410, -3.8981] 
    cap_w = np.array(cap_w)

    cap_name = ['black', 'white', 'puppy', 'in', 'the', 'middle', 'is', 'eating', 'food', 'in', 'man\'s','hand']
    plt.show()
    bar_name = './sample/sample_15.pdf' 
    plt.savefig(bar_name)


if  __name__ == '__main__':
    visualize_caption_weight_v3()
