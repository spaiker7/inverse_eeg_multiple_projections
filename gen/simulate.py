from pathlib import Path
import yaml
from tqdm import tqdm

import mne
import numpy as np

class Simulation:
	''' Simulate mean activity in labels specified by provided parcellation.
	
	Parameters
	----------
	settings: dict
		The Settings for the simulation. Keys:

		num_active_labels: int/tuple/list
			number of active labels. Can be a single number or a list of two numbers specifying a random number from range.
		eeg_snr:
			SNR(dB) = 10 * log10 (A_signal / A_noise)
			The desired average SNR of the electrodes

	'''
	def __init__(
			self, fwd, subjects_dir, 
			parcellation, save_dir, settings
			):
		
		self.settings = settings
		self.subjects_dir = subjects_dir
		self.save_dir = save_dir
		
		# fix the dipoles' orientations
		if not fwd['surf_ori']:
			self.fwd = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, 
														use_cps=True, verbose=0)

		self.subject = self.fwd['src'][0]['subject_his_id']

		self.labels = mne.read_labels_from_annot(self.subject, parc=parcellation, hemi='both', surf_name='white', 
												  subjects_dir=self.subjects_dir, sort=True,
												  annot_fname=None, regexp=None)
		
		self.leadfield = self.fwd['sol']['data']
		print("Leadfield size : %d sensors x %d dipoles" % self.leadfield.shape)

		source_model = self.fwd['src']
		self.dip_vertices = [source_model[0]['vertno'], source_model[1]['vertno']]
		self.dip_pos = [source_model[0]['rr'][source_model[0]['vertno']],
							source_model[1]['rr'][source_model[1]['vertno']]]


	def simulate(self, n_samples):
		''' simulate sources and EEG data'''
		self.n_samples = n_samples

		n_labels = self.settings['num_active_labels']
		n_labels_name = '-'.join(map(str, n_labels))
		sim_type_name = f"Parcellation_{n_labels_name}-lbl"

		sensors_save_to = Path(self.save_dir) / sim_type_name / "sensors"
		sensors_save_to.mkdir(parents=True, exist_ok=True)

		dipoles_save_to = Path(self.save_dir) / sim_type_name / "dipoles"
		dipoles_save_to.mkdir(parents=True, exist_ok=True)

		file_eeg_name = "eeg-{}_snr-{}.npy"
		file_dip_name = "dip-{}_lbls-{}.npy"

		for i in tqdm(range(n_samples)):
			source_data, lbls = self.simulate_label_source(n_labels=n_labels)
			eeg_data, snr = self.simulate_eeg(source_data, eeg_snr=self.settings['eeg_snr']) 
			
			fname_dip = Path(dipoles_save_to) / file_dip_name.format(i, lbls)
			np.save(fname_dip, source_data)
			
			fname_eeg = Path(sensors_save_to) / file_eeg_name.format(i, snr)
			np.save(fname_eeg, eeg_data)
				


	def simulate_eeg(self, source_data, eeg_snr, scale=1e-4):

		eeg = np.matmul(self.leadfield, source_data)*scale
		
		if eeg_snr:

			snr = np.random.randint(*eeg_snr)

			A_signal = np.mean(np.abs(eeg))
			A_noise = A_signal / (10**(snr/10))
			noise = np.random.normal(0, A_noise, len(eeg))
			noise = np.expand_dims(noise, axis=1)
			eeg_noised = np.copy(eeg) + noise

			return eeg_noised, snr
		else:
			return eeg

	def simulate_label_source(self, n_labels):
		'''pick vertices in a label and randomly set current values'''


		N_labels = len(self.labels)
		label_list=[]

		_, n_dipoles = self.leadfield.shape
		data = np.zeros((n_dipoles,1))

		lbls = np.random.randint(n_labels[0], n_labels[1])
		chosen_label_ids = np.random.randint(0, N_labels, lbls)

		for lbl_id in chosen_label_ids:
			lbl_cur = self.labels[lbl_id]
			label_list.append(lbl_cur)

			# get indices to assign data[lbl_id]
			hemi = lbl_cur.name.split("-")[-1]
			if hemi=='rh':
				n_dipoles_lh = self.dip_vertices[0].shape
				v = lbl_cur.get_vertices_used(vertices=self.dip_vertices[1])
				vert = self.dip_vertices[1]
				v_stc_id = np.nonzero(np.in1d(vert, v))[0]
				v_stc_id += n_dipoles_lh
			else:
				vert = self.dip_vertices[0]
				v = lbl_cur.get_vertices_used(vertices=self.dip_vertices[0])
				v_stc_id = np.nonzero(np.in1d(vert, v))[0]

			# generate the sources magnitudes as the gaussian distribution inside the label
			mean = np.random.uniform(0.25, 0.97)
			var_scale = np.random.uniform(0.005, 0.03)
			current_values = np.random.normal(loc=mean, scale=var_scale, size=(len(v_stc_id),1))
		   

			data[v_stc_id,:] = current_values

		return data, lbls


if __name__ == "__main__":
	
	with open('subjects_config.yml', 'r') as config_file:
			config = yaml.safe_load(config_file)

	fwd_template = "{subj}_{eeg_montage}-dip_{spacing_ico}-bem_fwd.fif"

	if not Path(config["gen_dir"]).exists():
		Path(config["gen_dir"]).mkdir(parents=True, exist_ok=True)
		
	for subject, sub_dict in config["subj_dict"].items():

		# eeg info
		eeg_path = Path(config["eeg_dir"]) / sub_dict["eeg_raw"]
		eeg_raw = mne.io.read_raw_fif(eeg_path, preload=True)
		eeg_raw.info["bads"] = []
		eeg_raw.pick_types(eeg=True).set_eeg_reference(projection=True)
		eeg_raw = eeg_raw.interpolate_bads(reset_bads=True, mode='accurate', origin='auto')
		print(f"eeg_path: {eeg_path}")

		# trans
		trans_path = Path(config["trans_dir"]) / sub_dict["trans"]
		trans = mne.read_trans(trans_path)
		print(f"trans_path: {trans_path}")

		# fwd
		fwd_fname = fwd_template.format(subj=subject,
										eeg_montage=config["eeg_montage"],
										spacing_ico=f'ico{config["spacing_ico"]}')
		
		fwd_path = Path(config["fwd_dir"]) / fwd_fname
		fwd = mne.read_forward_solution(fwd_path)
		print(f"fwd_path: {fwd_path}")
		
		print(f"\n{subject} files read")

		# simulate data
		save_dir = Path(config["gen_dir"]) / fwd_fname.replace(".fif","/")
	
		sim = Simulation(fwd, config["subjects_dir"], config["parcellation"],
									save_dir=save_dir, settings=config["sim_settings"])
		sim.simulate(n_samples=config["n_samples"])

		print(f'\n {config["n_samples"]} eeg-dipoles paired samples generated to {save_dir}')