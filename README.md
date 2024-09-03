## Inverse eeg with projections
The repository provides synthetic generation and representation of neural activity as images that could be used for training NN based inverse solvers. The code is mainly based on [mne](https://mne.tools/stable/index.html) python package. The individuals' MRI structural data is used to create forward solutions and then morphed to uniform average brain/head anatomy. To simulate neural activity patterns, [Schaefer's atlas](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal) is used to parcellate the cerebral cortex surface into 200 regions, any parcellation could be specified as you wish. In random number of specified regions we generate mean magnitude.

For sensors, 5 hyperplanes are applied: left and right lateral, posterior, anterior, and superior. After being projected, the sensors' coordinates are downscaled to the chosen grid space (256x256) and linearly interpolated to create topographic maps.

<img src="https://github.com/spaiker7/inverse_eeg_with_projections/assets/70488161/e2c96e11-db6d-48ce-9648-7935aa43032f" width=100% height=100%> 

##
UNetCBAM was adapted for regression task and trained as a benchmark model on 4 different subjects' morphed anatomy to predict activity in 7 cortical views.

<img src="https://github.com/user-attachments/assets/ee4e77f6-c134-4f50-a393-a90608a8315a" width=100% height=100%> 

## Generate data
To generate your own dataset, you need the following preprocessed MRI data from [freesurfer](https://surfer.nmr.mgh.harvard.edu/) for each subject:

1. **BEM** (Boundary Element Model): model used to simulate the electrical properties of the brain.
2. **surf**: surface models of the brain, which are used to define the boundaries for the BEM model.
3. **label**: annotations files for different brain regions (parcellation).
4. **trans**: transformation matrix that maps the MRI data to the surface models.

Additionally, you need to provide raw EEG files with montages. This information should be specified in the gen/subjects_config.yml file. This file is used to configure the generation of synthetic activity for each subject. It includes details such as the paths to the required files and the specific settings for the generation process.

execute these scripts in following order:

_**compute_fwd_and_morph -> simulate -> project_and_interpolate**_


## Use pretrained model
If you want to use pre-trained Attention-Unet to predict activity from your own EEG or to fine-tune - follow the below code:

```python
import torch
from gen.project_and_interpolate import CorticalProjectionPreprocessor
from model.models import UNetCBAM
from plot import plot_projections

projector = CorticalProjectionPreprocessor('subjects_config.yml')
topomaps = projector.project_and_interpolate_sensors()

model = UNetCBAM()
checkpoint = torch.load(args_dict['checkpoint'])
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

predicted_activity = model(topomaps)
plot_projections(predicted_activity)

```

## Convert back to source space
If you need to transforms projections back to source estimate, we prodive converter that generates masks of each dipole in each projection. The mean dipole magnitude calculated by the model predictions:

```python
from gen.convert_stc import SourceEstimateConverter
```

