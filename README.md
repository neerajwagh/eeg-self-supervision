<!-- # eeg-self-supervision
Resources for the paper titled "Domain-guided Self-supervision of EEG Data Improves Downstream Classification Performance and Generalizability". Accepted at ML4H Symposium 2021 with an oral spotlight! -->

# Domain-guided Self-supervision of EEG Data Improves Downstream Classification Performance and Generalizability

_*Authors*: Neeraj Wagh, Jionghao Wei, Samarth Rawal, Brent Berry, Leland Barnard, Benjamin Brinkmann, Gregory Worrell, David Jones, Yogatheesan Varatharajah_

_*Affiliation*:University of Illinois at Urbana-Champaign, Mayo Clinic_

## Work accepted in proceedings of the ML4H Symposium 2021 with an oral spotlight!

<!-- - ArXiv Pre-print: <> -->
<!-- - PMLR Paper: <> -->
<!-- - ML4H Poster: <>
- ML4H 10-minute Video: <>
- ML4H Slides: <>
- Code: [GitHub Repo]() -->
- Camera Ready Version: <https://drive.google.com/file/d/1hUB_WPYLTUusqJaUanSOxLjD3OD8NkQ1/view?usp=sharing>
- Final Models, Pre-computed Features, Training Metadata: [Box](https://uofi.box.com/s/80lygevy4d7mc3nwne8267wlkcjiun0v)
- Raw Data: [MPI LEMON](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) (no registration needed), [TUH EEG Abnormal Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/) ([needs registration](https://www.isip.piconepress.com/projects/tuh_eeg/html/request_access.php))

### Mapping naming conventions between the paper and code 
- This section will clear the confusion about naming of models and taskes.
- Models: 
    - FULLSSL refers to "HS+BSE+AC" in the paper.
    - DBRTriplet refers to "BSE+AC".
    - BTTriplet refers to "HS+AC".
    - BTDBR refers to "HS+BSE".
    - Triplet refers to "AC only".
    - DBR refers to "BSE only".
    - BT refers to "HS only".
    - SOTA refers to "ShallowNet".
- Tasks:
    - "EEG Grade" is the "condition" task with TUH dataset.
    - "Eye state" is the "condition" task with Lemon dataset.
- Choices for tasks: "condition", "gender", "age".
- Choices for dataset: "lemon", "tuh".
- Choices for ablation models: "FULLSSL", "DBRTriplet", "BTTriplet", "BTDBR", "Triplet", "DBR", "BT".
### How to Reproduce the Results Reported in the Paper
1. Download _finetuned_models_ folder, pre-computed feature arrays(power spectral density, topo data, and preproc timeseries), training metadata from [Box](https://uofi.box.com/s/80lygevy4d7mc3nwne8267wlkcjiun0v).
2. Place _finetuned_models_ folder, all computed feature arrays, and metadata files into _evaluation_ folder. The code expects these files to be present in _evaluation_ folder.
3. Enter _evaluation_ folder.
4. To evaluate linear baseline, run the following command:
```python
python linear_baseline_eval.py --dataset={wanted_dataset} --task={wanted_task}
```
5. To evaluate proposed ablation models, run the following command:
```python
python ablation_models_eval.py --gpu_idx={desired gpu index} --dataset={wanted dataset} --task={wanted task} --mode={desired ablation model}
```
6. To evaluate SOTA model, run the following command:
```python
python SOTA_eval.py --gpu_idx={desired gpu index} --dataset={wanted dataset} --task={wanted task} 
```
### How to Fine-tune Existing Pre-trained Models for Downstream Tasks
1. Download _pretrained_models_ folder, pre-computed feature arrays, training metadata from [Box](https://uofi.box.com/s/80lygevy4d7mc3nwne8267wlkcjiun0v).
2. Place _pretrained_models_ folder, all computed feature arrays, and metadata files into _Fine-tune_ folder. The code expects these files to be present in _Fine-tune_ folder.
3. Enter _Fine-tune_ folder.
4. To fine-tune ablation models, run the following command: 
```python
python ablation_pipeline.py --gpu_idx={desired gpu index} --dataset={wanted dataset} --task={wanted task} --mode={desired ablation model}
```
5. To fine-tune SOTA mode, run the following command:
```python
python SOTA_pipeline.py --gpu_idx={desired gpu index} --dataset={wanted dataset} --task={wanted task}
```
### How to Perform Pre-training using Domain-guided Self-superivsed Tasks
1. Enter _Pretrain_ folder.
2. Run the following command:
```python
python pretrain.py --gpu_idx={desired gpu index} --mode={Wanted ablation model}
```

### Contact

- Issues regarding non-reproducibility of results or support with the codebase should be emailed to _nwagh2@illinois.edu_
- Neeraj: nwagh2@illinois.edu / [Website](http://neerajwagh.com/) / [Twitter](https://twitter.com/neeraj_wagh) / [Google Scholar](https://scholar.google.com/citations?hl=en&user=lCy5VsUAAAAJ)
- John: wei33@illinois.edu
- Yoga: varatha2@illinois.edu / [Website](https://sites.google.com/view/yoga-personal/home) / [Google Scholar](https://scholar.google.com/citations?user=XwL4dBgAAAAJ&hl=en)

<!-- ### Citation -->