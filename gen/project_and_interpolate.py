from tqdm import tqdm
from pathlib import Path

import yaml
import h5py
import numpy as np
import mne
import cv2

from scipy.interpolate import griddata

mne.viz.set_3d_backend('pyvistaqt')

from utils import crop_img_by_zeros

class CorticalProjectionPreprocessor:
    """
    Generates topomap(s) by projecting and linearly interpolating sensor data onto a 2d grid.
    Projects sources onto a cortical inflated surface and creates brain plots in specified views (mne).
    Calculates loss masks for the fsaverage cortical surface to be used during the training of models.
    Creates h5 archive for the convinience of training and storage.

    Parameters:
    - subjects_config: .yml file        
    - grid: int
        The input-output images resolution. default=256
    - onto_morph: bool
        Whether to project sources on fsaverage preprocessed morphed brain surface and electrodes onto the head.
        if False - project onto the subject's surface.

    WARNING:
        self.project_sources_onto_cortex() constanly creates pop-up windows to preprocess images.
        In mne 1.7.0 version offscreen rendering is not recommended.

    """
    def __init__(self,
                 subjects_config,
                 grid=256,
                 onto_morph=True): 
        
        self.grid = grid
        self.onto_morph = onto_morph

        with open(subjects_config, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
        
        if self.onto_morph:

            fsavg_trans_path = Path(self.config["trans_dir"]) / 'fsaverage-trans.fif'
            fsavg_trans = mne.read_trans(fsavg_trans_path)

            sub_dict = next(iter(self.config["subj_dict"].values()))
            eeg_path = Path(self.config["eeg_dir"]) / sub_dict["eeg_raw"]
            eeg_raw = mne.io.read_raw_fif(eeg_path, preload=True)
            eeg_raw.info["bads"] = []
            eeg_raw.pick_types(eeg=True).set_eeg_reference(projection=True)
            surf_path = Path(self.config["subjects_dir"]) / 'fsaverage' / 'bem/outer_skin.surf'
            surf = mne.surface.read_surface(surf_path, return_dict=True)[-1]
            eeg_pos = np.array([eeg_raw.info['chs'][k]['loc'][:3] for k in range(eeg_raw.info['nchan'])])
            eeg_pos_trans = mne.transforms.apply_trans(fsavg_trans, eeg_pos)
            self.fsavg_eeg_pos, _ = mne.surface._project_onto_surface(eeg_pos_trans*1000, surf, 
																project_rrs=True, return_nn=True)[2:4]

    def project_data(self):


        self.fwd_template = "{subject}_{eeg_montage}-dip_{spacing_ico}-bem_fwd.fif"
        self.n_labels = '-'.join(map(str, self.config["sim_settings"]["num_active_labels"]))       

        self.hemi_view_combinations = [['both', 'dorsal'], ['both', 'caudal'], ['both', 'rostral'],
                                ['lh', 'lateral'], ['lh', 'medial'], ['rh', 'lateral'], ['rh', 'medial']]
        
        num_subjects = len(list(self.config["subj_dict"].keys()))

        self.in_archive_name = Path(self.config["gen_dir"]) / \
            f'sensors_dipoles_subj-{num_subjects}_ico{self.config["spacing_ico"]}_lbl-{self.n_labels}.h5'
        self.out_archive_name = Path(self.config["gen_dir"]) / \
            f'topomaps_cortex-views_subj-{num_subjects}_ico{self.config["spacing_ico"]}_lbl-{self.n_labels}.h5'

        self.fwds = {}

        for subject in self.config["subj_dict"].keys():

            fwd_fname = self.fwd_template.format(subject=subject,
                                                eeg_montage=self.config["eeg_montage"],
                                                spacing_ico=f'ico{self.config["spacing_ico"]}')
            fwd_path = Path(self.config["fwd_dir"]) / fwd_fname

            self.fwds[subject] = mne.read_forward_solution(fwd_path)
            

        if self.onto_morph:

            self.loss_masks = get_cortical_loss_masks(self.config, self.hemi_view_combinations)

        with h5py.File(self.in_archive_name, 'r') as in_file, \
            h5py.File(self.out_archive_name, 'a') as out_file:
            
            for subj in self.config["subj_dict"].keys():
                
                if subj not in out_file['cortex-views'].keys():
                    out_file.create_group(f'cortex-views/{subj}')
                if subj not in out_file['topomaps'].keys():
                    out_file.create_group(f'topomaps/{subj}')

                fwd = self.fwds[subj]
                src = fwd['src']
                vertices = [src[0]['vertno'], src[1]['vertno']]

                print(f"\nGenerating topomaps and cortical views for {subj} subject:")

                for sensors_name, dipoles_name in tqdm(list(zip(list(in_file[f'sensors/{subj}']), list(in_file[f'dipoles/{subj}'])))):
                    
                    topomaps_name = '-'.join(sensors_name.split('-')[1:])
                    cortex_views_name = '-'.join(dipoles_name.split('-')[1:])

                    if topomaps_name not in out_file['topomaps'][subj].keys():
                        topomaps = self.project_and_interpolate_sensors(in_file, subj, sensors_name)
                        out_file.create_dataset(f'topomaps/{subj}/{topomaps_name}', data=topomaps,
                                            shape=topomaps.shape, dtype=np.float16, compression="gzip", compression_opts=5)
                    
                    if cortex_views_name not in out_file['cortex-views'][subj].keys():
                        cortex_views = self.project_sources_onto_cortex(in_file, subj, dipoles_name, vertices)
                        out_file.create_dataset(f'cortex-views/{subj}/{cortex_views_name}', data=cortex_views,
                                        shape=cortex_views.shape, dtype=np.float16, compression="gzip", compression_opts=5)
                        
        print("Done!")


    def project_and_interpolate_sensors(self, in_file, subj, sensors_name, sep_hyperplanes_coords=[0, -20, 0]):

        sensors = np.array(in_file[f'sensors/{subj}/{sensors_name}'])
        x, y, z = sep_hyperplanes_coords

        if self.onto_morph:
            sensors = np.hstack((self.fsavg_eeg_pos, sensors))

        horizontal = sensors[sensors[:, 2]>= z]
        horizontal = np.delete(horizontal, 2, 1)

        frontal = sensors[sensors[:, 1] >= y]
        frontal = np.delete(frontal, 1, 1)

        back = sensors[sensors[:, 1] <= y]
        back = np.delete(back, 1, 1)

        left = sensors[sensors[:, 0] <= x]
        left = np.delete(left, 0, 1)

        right = sensors[sensors[:, 0] >= x]
        right = np.delete(right, 0, 1)

        topomaps_names = ['horizontal', 'frontal', 'back', 'left', 'right']
        data = [horizontal, frontal, back, left, right]
        topomaps = {}

        for projection, data in zip(topomaps_names, data):
            x_grid = np.linspace(min(data[:, 0]), max(data[:, 0]), self.grid)
            y_grid = np.linspace(min(data[:, 1]), max(data[:, 1]), self.grid)
            X, Y = np.meshgrid(x_grid, y_grid)
            topomaps[projection] = griddata(data[:,:2], data[:,2], (X, Y), method='linear')

        topomaps['horizontal'] = cv2.flip(cv2.rotate(topomaps['horizontal'], cv2.ROTATE_180), 1)
        topomaps['back'] = cv2.flip(cv2.rotate(topomaps['back'], cv2.ROTATE_180), 1)
        topomaps['left'] = cv2.rotate(topomaps['left'], cv2.ROTATE_180)
        topomaps['right'] = cv2.flip(cv2.rotate(topomaps['right'], cv2.ROTATE_180), 1)
        topomaps['frontal'] = cv2.rotate(topomaps['frontal'], cv2.ROTATE_180)

        topomaps = np.nan_to_num(np.stack(topomaps.values()))
        topomaps = np.array(topomaps, dtype=np.float16)

        return topomaps


    def project_sources_onto_cortex(self, in_file, subj, dipoles_name, vertices):
        
        subject_name = self.config["subj_dict"][subj]["fs_name"]
        sources = np.array(in_file[f'dipoles/{subj}/{dipoles_name}'])
        stc = mne.SourceEstimate(sources, vertices, subject=subject_name, tmin=0, tstep=1/1000)

        if self.onto_morph:
            morph_path = Path(self.config["morph_dir"]) / \
                f'{subject_name}-fsaverage-ico{self.config["spacing_ico"]}.h5'
            morph = mne.read_source_morph(morph_path)
            stc = morph.apply(stc)

        views = []

        Path('./tmp').mkdir(parents=True, exist_ok=True)
        for hemi_view_combination, loss_mask in zip(self.hemi_view_combinations, self.loss_masks):

            brain = stc.plot(
                subjects_dir=self.config["subjects_dir"],
                initial_time=0,
                hemi=hemi_view_combination[0],
                size=(500, 500),
                smoothing_steps=4,
                time_viewer=False,
                clim={'kind': 'value', 'pos_lims':(0., 500.0, 1000.0)},
                colorbar=False,
                colormap='gray',
                cortex=(0, 0, 0),
                views=hemi_view_combination[1],
                )

            screenshot = brain.screenshot()
            brain.close()

            tmp_path = 'tmp/' + '_'.join(hemi_view_combination) + '.png'
            cv2.imwrite(tmp_path, screenshot)

            view = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
            view = crop_img_by_zeros(view, by_img=loss_mask)
            view = cv2.resize(view, (self.grid,self.grid))

            views.append(view)

        views = np.array(views, dtype=np.float16)/255*100

        return views


def get_cortical_loss_masks(config, hemi_view_combinations, subject='fsaverage', save_dir='', 
                            fwd=None, save=True, grid=256):
    """
    Calculates cortcial projections' masks (hemi view combinations).
    """

    if subject == 'fsaverage':
        save_dir = Path(config["gen_dir"])
        fname_fsavg_src = Path(config["subjects_dir"]) / "fsaverage/bem/fsaverage-ico-5-src.fif"
        src = mne.read_source_spaces(fname_fsavg_src)
        vertices = [src[0]['vertno'], src[1]['vertno']]
        stc = mne.SourceEstimate(np.ones(src[0]['nuse']*2), vertices, subject=subject, tmin=0, tstep=1/1000)
    else:
        subject_name = config["subj_dict"][subject]["subj"]
        vertices = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
        stc = mne.SourceEstimate(np.ones(fwd['nsource']), vertices, subject=subject_name, tmin=0, tstep=1/1000)

    loss_masks = []
    uncroppped_loss_masks = []

    Path('./tmp').mkdir(parents=True, exist_ok=True)
    for hemi_view_combination in hemi_view_combinations:
        brain = stc.plot(
            subjects_dir=config["subjects_dir"],
            initial_time=0,
            hemi=hemi_view_combination[0],
            size=(500, 500),
            smoothing_steps=3,
            time_viewer=False,
            clim={'kind': 'value', 'pos_lims':(0.01, 0.6, 1)},
            colorbar=False,
            colormap='gray',
            cortex=(1, 1, 1),
            views=hemi_view_combination[1],
            )

        screenshot = brain.screenshot()
        brain.close()

        tmp_path = 'tmp/' + '_'.join(hemi_view_combination) + '.png'
        cv2.imwrite(tmp_path, screenshot)

        view = cv2.imread(tmp_path, cv2.IMREAD_GRAYSCALE)
        view[view>0] = 1

        cropped_view = crop_img_by_zeros(view)

        loss_masks.append(cv2.resize(cropped_view, (grid,grid)))
        uncroppped_loss_masks.append(view)

    # uncropped loss masks needed for matching mne.plot output images resolution
    # to generate cortical projections. The cropped ones utilized for training models.
    loss_masks = np.array(np.stack(loss_masks), dtype=np.float32)
    uncroppped_loss_masks = np.array(np.stack(uncroppped_loss_masks), dtype=np.float32)

    if save:
        np.savez_compressed((save_dir / f'loss_masks_{subject}'), loss_masks)

    return uncroppped_loss_masks

if __name__ == "__main__":
    data_processor = CorticalProjectionPreprocessor('subjects_config.yml')
    data_processor.project_data()
