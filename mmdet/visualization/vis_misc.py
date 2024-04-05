### visual func
import numpy as np
import matplotlib.pyplot as plt

def show_conf(class_conf):
    class_conf_np = class_conf.squeeze().cpu().detach().numpy()
    mask = np.argmax(class_conf_np, axis=0)
    plt.imshow(mask, cmap="jet", alpha=0.5)
    plt.show()
    
def show_img(img):
    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_np)
    plt.show()
    
def show_center(center_map):
    center_map_numpy = center_map.cpu().detach().squeeze().numpy()
    plt.imshow(center_map_numpy, cmap='gray')
    plt.axis('off')
    plt.show()