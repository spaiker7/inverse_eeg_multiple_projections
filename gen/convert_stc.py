from os import listdir
from tqdm import tqdm
from pathlib import Path

import yaml
import h5py
import numpy as np
import mne
import cv2

from project_and_interpolate import get_cortical_loss_masks
from utils import norm_data, crop_img_by_zeros


class SourceEstimateConverter:
    """
    Generates projections of each dipole in the mne.SourceSpace on inflated cortex and extracts them as masks
    to calculate mean intensity in each projection of the dipole by views predicted by model.

    Parameters:
    - subjects_config: .yml file
    - grid: int
        The input-output images resolution. default=256
    - morph_preds: bool
        If True - fsaverage brain will be used to preprocess the predicted images
        else: original source space has to be provided

    WARNING:
    func generate_dipoles_masks() constanly creates pop-up windows to preprocess images.
    In mne 1.7.0 version offscreen rendering is not recommended.

    """
    def __init__(self,
                 subject='fsaverage',
                 morph_preds=True,
                 subjects_config='subjects_config.yml',
                 grid=256,
                 ):

        self.hemi_view_combinations = [['both', 'dorsal'], ['both', 'caudal'], ['both', 'rostral'],
                                       ['lh', 'lateral'], ['lh', 'medial'], ['rh', 'lateral'], ['rh', 'medial']]
        
        self.grid = grid
        self.morph_preds = morph_preds

        with open(subjects_config, 'r') as config_file:
            self.config = yaml.safe_load(config_file)

        
        if self.morph_preds:

            src_fsavg_path = Path(self.config['subjects_dir']) / 'fsaverage' / 'src' / f'fsaverage-ico{self.config["spacing_ico"]}-src.fif'
            if not src_fsavg_path.exists():
                self.src_fsavg = mne.setup_source_space('fsaverage',
                                spacing=f'ico{self.config["spacing_ico"]}',
                                subjects_dir=self.config["subjects_dir"],
                                add_dist=True, n_jobs=-1)
                self.src_fsavg.save(src_fsavg_path)
            else:
                self.src_fsavg = mne.read_source_spaces(src_fsavg_path)
            
            self.vertices = [self.src_fsavg[0]['vertno'], self.src_fsavg[1]['vertno']] 
            self.num_dip = self.src_fsavg[0]['nuse'] + self.src_fsavg[1]['nuse']

            self.save_dir = Path(self.config["gen_dir"]) / 'fsaverage_dipoles_masks'
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        else: 

            self.fwd_template = "{subj}_{eeg_montage}-dip_{spacing_ico}-bem_fwd.fif"
            self.n_labels_name = '-'.join(map(str, self.config["sim_settings"]["num_active_labels"]))

            fwd_fname = self.fwd_template.format(subj=subject,
                                                eeg_montage=self.config["eeg_montage"],
                                                spacing_ico=f'ico{self.config["spacing_ico"]}')
            fwd_path = Path(self.config["fwd_dir"]) / fwd_fname
            self.fwd = mne.read_forward_solution(fwd_path)

            self.save_dir = Path(self.config["gen_dir"]) / fwd_fname.replace(".fif","") / f'Parcellation_{self.n_labels_name}-lbl'
        

    def generate_dipoles_masks(self):

        if self.morph_preds:
            
            loss_masks = get_cortical_loss_masks(self.config, self.hemi_view_combinations,
                                                 save=False)

            for dip_id in tqdm(range(len(list(self.save_dir.iterdir())), self.num_dip)):
                data = np.zeros(self.num_dip)
                data[dip_id] = 1
                stc = mne.SourceEstimate(data, self.vertices, subject='fsaverage', tmin=0, tstep=1/1000)
                self.get_single_dipole_mask_on_cortex(dip_id, stc, 'fsaverage', self.save_dir, loss_masks)

        # else:

        #     for subject in self.config["subj_dict"].keys():

        #         src = self.fwd["src"]
        #         vertices = [src[0]['vertno'], src[1]['vertno']]

        #         loss_masks = get_cortical_loss_masks(self.config, self.hemi_view_combinations,
        #                                              subject, save_dir=self.save_dir, fwd=self.fwd)
  
        #         save_dir = self.save_dirs[subject] / 'dipoles_masks'
        #         self.save_dir.mkdir(parents=True, exist_ok=True)

        #         num_dip = src[0]['nuse'] + src[1]['nuse']
        #         for dip_id in tqdm(range(num_dip)):
        #             data = np.zeros(num_dip)
        #             data[dip_id] = 1
        #             stc = mne.SourceEstimate(data, vertices, subject=subject, tmin=0, tstep=1/1000)
        #             self.get_single_dipole_mask_on_cortex(dip_id, stc, subject, save_dir, loss_masks)


    def get_single_dipole_mask_on_cortex(self, dip_id, stc, subject, save_dir, loss_masks):
        """
        Calculates cortcial dipole projections' masks (hemi view combinations).
        """

        dipole_masks = []

        Path('./temp').mkdir(parents=True, exist_ok=True)
        for hemi_view_combination, loss_mask in zip(self.hemi_view_combinations, loss_masks):
            brain = stc.plot(
                subjects_dir=self.config["subjects_dir"],
                initial_time=0,
                hemi=hemi_view_combination[0],
                size=(500, 500),
                smoothing_steps=3,
                time_viewer=False,
                clim={'kind': 'value', 'pos_lims':(0.01, 0.6, 1)},
                colorbar=False,
                colormap='gray',
                cortex=(0, 0, 0),
                views=hemi_view_combination[1],
                )

            screenshot = brain.screenshot()
            brain.close()

            temp_path = 'temp/' + '_'.join(hemi_view_combination) + '.png'
            cv2.imwrite(temp_path, screenshot)

            dipole_mask = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            dipole_mask[dipole_mask>0] = 1
            dipole_mask = crop_img_by_zeros(dipole_mask, by_img=loss_mask)

            dipole_masks.append(cv2.resize(dipole_mask, (self.grid,self.grid)))

        dipole_masks = np.array(np.stack(dipole_masks), dtype=np.float32)
        np.savez_compressed(Path(save_dir / f'{subject}_dipole_mask_{dip_id}'), dipole_masks)


    def archive_to_source_estimates(self, archive_path, dip_masks_dir, save_path='', compute_trues=False):
        
        dataset = h5py.File(archive_path, 'r')
        keys = list(dataset.keys())

        val_predicts_batches = []
        for key in keys:
            if key.split('_')[:2] == ['val', 'output']:
                val_predicts_batches.append(dataset[key])

        val_predicts = val_predicts_batches[0]
        for batch in val_predicts_batches[1:]:
            val_predicts = np.concatenate((val_predicts, batch))
        
        self.convert_imgs_to_source_estimates(val_predicts, dip_masks_dir, save_path)
        
        if compute_trues:

            val_trues_batches = []
            for key in keys:
                if key.split('_')[:2] == ['val', 'true']:
                    val_trues_batches.append(dataset[key])

            val_trues = val_trues_batches[0]
            for batch in val_trues_batches[1:]:
                val_trues = np.concatenate((val_trues, batch))
            
            save_trues_path = save_path.parent / '_'.join(save_path.stem.split('_')[:3])
            self.convert_imgs_to_source_estimates(val_trues, dip_masks_dir, save_trues_path)

        

    def convert_imgs_to_source_estimates(self, input_views, dip_masks_dir, save_path='', normalize=False):
        """
        Input views format: (n_times, hemi_view_combinations, grid, grid)
        """
        n_dipoles = len(list(dip_masks_dir.iterdir()))
        n_times = len(input_views)
        data = np.zeros((n_dipoles, n_times))

        if normalize:
            input_views = norm_data(input_views)

        for dip_masks_path in tqdm(dip_masks_dir.iterdir()):
            dip_masks = np.load(dip_masks_path)["arr_0"]
            
            dip_id = int(Path(dip_masks_path).stem.split('_')[-1])
            for timestamp, views in enumerate(input_views):
                dip_views = dip_masks * views

                if np.any(dip_views[dip_views>0]):
                    data[dip_id, timestamp] = np.mean(dip_views[dip_views>0])
                else:
                    data[dip_id, timestamp] = 0

        if self.morph_preds:
            stc = mne.SourceEstimate(data, self.vertices, subject='fsaverage', tmin=0, tstep=1/1000)

        if save_path:
            stc.save(save_path)

        return stc
    
if __name__ == "__main__":

    converter = SourceEstimateConverter()
    converter.generate_dipoles_masks()