from pathlib import Path
import yaml
import mne

with open('subjects_config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)

fwd_template = "{fs_name}_{eeg_montage}-dip_{ico_spacing}-bem_fwd.fif"

fname_fsaverage_src = Path(config["subjects_dir"]) / "fsaverage/bem/fsaverage-ico-5-src.fif"
src_to = mne.read_source_spaces(fname_fsaverage_src)

for subject, sub_dict in config["subj_dict"].items():
        
    src = mne.setup_source_space(sub_dict["fs_name"], spacing=f'ico{config["spacing_ico"]}', subjects_dir=config["subjects_dir"], n_jobs=-1)
    model = mne.make_bem_model(sub_dict["fs_name"], ico=config["spacing_ico"], subjects_dir=config["subjects_dir"])   
    bem = mne.make_bem_solution(model)

    eeg_path = Path(config["eeg_dir"]) / sub_dict["eeg_raw"]
    eeg_raw = mne.io.read_raw_fif(eeg_path, preload=True)
    eeg_raw.info["bads"] = []
    eeg_raw.pick_types(eeg=True).set_eeg_reference(projection=True)
    eeg_raw=eeg_raw.interpolate_bads(reset_bads=True, mode='accurate', origin='auto')

    trans_path = Path(config["trans_dir"]) / sub_dict["trans"]

    fwd = mne.make_forward_solution(
        eeg_raw.info,
        trans=trans_path,
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=-1,
        verbose=True
    )

    fwd_fname = fwd_template.format(subj=subject,
                                    eeg_montage=config["eeg_montage"],
                                    ico_spacing=f'ico{config["spacing_ico"]}')
    fwd_path = Path(config["fwd_dir"]) / fwd_fname
    mne.write_forward_solution(fwd_path, fwd)

    morph = mne.compute_source_morph(
        fwd["src"],
        subject_from=sub_dict["fs_name"],
        subject_to="fsaverage",
        src_to=src_to,
        subjects_dir=config["subjects_dir"]
    )
    morph.save(Path(config["morph_dir"]) / f'{sub_dict["fs_name"]}-fsaverage-ico{config["spacing_ico"]}.h5')