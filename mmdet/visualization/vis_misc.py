### visual func
import numpy as np
import matplotlib.pyplot as plt
from mmengine.structures.instance_data import InstanceData
import torch
from sklearn.manifold import TSNE

def show_conf(class_conf):
    class_conf_np = class_conf.squeeze().cpu().detach().numpy()
    mask = np.argmax(class_conf_np, axis=0)
    plt.figure()
    plt.imshow(mask, cmap="jet", alpha=0.5)
    plt.show()
    
def show_img(img):
    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.figure()
    plt.imshow(img_np)
    plt.show()
    
def show_center(center_map):
    center_map_numpy = center_map.cpu().detach().squeeze().numpy()
    plt.figure()
    plt.imshow(center_map_numpy, cmap='gray')
    plt.axis('off')
    plt.show()
    
def show_TSNE(output: dict):
    tsne = TSNE(random_state=0, perplexity =50)
    tsne_output= tsne.fit_transform(output['embedding'].cpu().detach().numpy())
    labels = output['labels'].cpu().detach().numpy()

    plt.figure(figsize=(16, 16))
    # plt.scatter(tsne_output[:, 0], tsne_output[:, 1], marker='o',c=labels, cmap='jet', alpha=0.5)

    selected_tsne_output = tsne_output[labels == 0]
    plt.scatter(selected_tsne_output[:, 0], selected_tsne_output[:, 1], marker='o', c='red', alpha=0.5)

    other_tsne_output = tsne_output[labels != 0]
    plt.scatter(other_tsne_output[:, 0], other_tsne_output[:, 1], marker='o', c='green', alpha=0.5)
    plt.show()

def add_to_dict(results_list: InstanceData, embedding: torch.tensor, data_dict: dict):
    score_thres = 0.4
    num_instances = torch.sum(results_list[0].scores > score_thres)
    
    labels = results_list[0].labels[:num_instances]
    embedding = embedding.squeeze()[:num_instances,:]
    
    if "labels" in data_dict:
        data_dict["labels"] = torch.cat((data_dict["labels"], labels))
    else:
        data_dict["labels"] = labels
        
    if "embedding" in data_dict:
        data_dict["embedding"] = torch.cat((data_dict["embedding"], embedding))
    else:
        data_dict["embedding"] = embedding
