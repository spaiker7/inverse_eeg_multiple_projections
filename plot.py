import numpy as np
from matplotlib import pyplot as plt

def plot_projections(imgs, dtype='cortical_views'):
    """
    type: str ['topomaps', 'cortical_views']
    """
    if dtype == 'topomaps':
        plt_names = ['horizontal', 'frontal', 'back', 'left', 'right']
    else:
        plt_names = ['dorsal', 'caudal', 'rostral', 'left lateral',
                     'left medial', 'right lateral', 'right medial']
        
    fig, axs = plt.subplots(1, len(imgs), figsize=(18, 3))
    for i in range(len(imgs)):
        axs[i].imshow(imgs[i], 'gray', vmin=np.min(imgs), vmax=np.max(imgs))
        axs[i].set_title(plt_names[i])
        axs[i].set_xticks([])
        axs[i].set_yticks([])

    cbar = fig.colorbar(axs[0].images[0], ax=axs, orientation='vertical', fraction=0.011)