<!-- # eeg-self-supervision
Resources for the paper titled "Domain-guided Self-supervision of EEG Data Improves Downstream Classification Performance and Generalizability". Accepted at ML4H Symposium 2021 with an oral spotlight! -->

# Domain-guided Self-supervision of EEG Data Improves Downstream Classification Performance and Generalizability

_*Authors*: Neeraj Wagh, Jionghao Wei, Samarth Rawal, Brent Berry, Leland Barnard, Benjamin Brinkmann, Gregory Worrell, David Jones, Yogatheesan Varatharajah_

_*Affiliation*: University of Illinois at Urbana-Champaign, Mayo Clinic_

## Work accepted in proceedings of the ML4H Symposium 2021 with an oral spotlight!

<!-- - ArXiv Pre-print: <> -->
<!-- - PMLR Paper: <> -->
<!-- - ML4H Poster: <>
- ML4H 10-minute Video: <>
- ML4H Slides: <>
- Code: [GitHub Repo]() -->
- Paper: <https://proceedings.mlr.press/v158/wagh21a.html>
- ML4H Poster: <https://drive.google.com/file/d/1-B7joEquzr4kqsfSGDqksUARWsxlXOyA/view?usp=sharing>
- ML4H 5-minute Video: <https://recorder-v3.slideslive.com/?share=56322&s=8950fb88-3f3f-4490-92ff-760b73e5a4b5>
- ML4H Slides: <https://drive.google.com/file/d/1uUma2e4GKIn8Zo7dEQRl_WkyVcBLzpaR/view?usp=sharing>
- Final Models, Pre-computed Features, Training Metadata: [Box](https://uofi.box.com/s/80lygevy4d7mc3nwne8267wlkcjiun0v)
- Raw Data: [MPI LEMON](http://fcon_1000.projects.nitrc.org/indi/retro/MPI_LEMON.html) (no registration needed), [TUH EEG Abnormal Corpus](https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_abnormal/) ([needs registration](https://www.isip.piconepress.com/projects/tuh_eeg/html/request_access.php))

### Installation
- A dockerfile is provided with all Python dependencies. 
### Mapping naming conventions between the paper and code 
- Models: 
    - FULLSSL in code refers to "HS+BSE+AC" in the paper.
    - DBRTriplet refers to "BSE+AC".
    - BTTriplet refers to "HS+AC".
    - BTDBR refers to "HS+BSE".
    - Triplet refers to "AC only".
    - DBR refers to "BSE only".
    - BT refers to "HS only".
    - SOTA refers to "ShallowNet".
- Tasks:
    - "EEG Grade" is the "condition" task with TUH dataset.
    - "Eye State" is the "condition" task with Lemon dataset.
### Command Line Arguments
- Input choices for tasks: "condition", "gender", "age".
- Input choices for dataset: "lemon", "tuh".
- Input choices for ablation models: "FULLSSL", "DBRTriplet", "BTTriplet", "BTDBR", "Triplet", "DBR", "BT".
### How to Reproduce the Results Reported in the Paper
1. Download _resources_ folder, which contains pre-computed feature arrays(power spectral density, topographical maps data, and preprocessed timeseries), metadata from [Box](https://uofi.box.com/s/80lygevy4d7mc3nwne8267wlkcjiun0v). Place it in the root directory of the project.
2. Enter _evaluation_ folder.
3. To evaluate linear baseline, run the following command:
```python
python linear_baseline_eval.py --dataset={dataset choice} --task={task choice}
```
4. To evaluate proposed ablation models, run the following command:
```python
python ablation_models_eval.py --gpu_idx={gpu index} --dataset={dataset choice} --task={task choice} --mode={ablation model choice}
```
5. To evaluate SOTA model, run the following command:
```python
python SOTA_eval.py --gpu_idx={gpu index} --dataset={dataset choice} --task={task choice} 
```
### How to Fine-tune Existing Pre-trained Models for Downstream Tasks
1. Download _resources_ folder from [Box](https://uofi.box.com/s/80lygevy4d7mc3nwne8267wlkcjiun0v). Place it in the root directory of the project.
2. Enter _Fine-tune_ folder.
3. To fine-tune ablation models, run the following command: 
```python
python ablation_pipeline.py --gpu_idx={gpu index} --dataset={dataset choice} --task={task choice} --mode={ablation model choice}
```
4. To fine-tune SOTA model, run the following command:
```python
python SOTA_pipeline.py --gpu_idx={gpu index} --dataset={dataset choice} --task={task choice}
```

### How to Perform Supervised Learning for Downstream Tasks
1. Download _resources_ folder from [Box](https://uofi.box.com/s/80lygevy4d7mc3nwne8267wlkcjiun0v). Place it in the root directory of the project.
3. Enter _supervised_learning_ folder.
4. To train the supervised learning models, run the following command:
```python
python linear_baseline_train.py --dataset={dataset choice} --task={task choice}
```
### [Work in Progress] - How to Perform Pre-training using Domain-guided Self-superivsed Tasks
<!-- 1. Enter _Pretrain_ folder.
2. Run the following command:
```python
python pretrain.py --gpu_idx={gpu index} --mode={wanted ablation model}
``` -->
### [Work in Progress] - How to Use Your Own Dataset to Train the Self-supervised Learning Tasks
### Contact

- Issues regarding non-reproducibility of results or support with the codebase should be emailed to Neeraj and John.
- Neeraj: nwagh2@illinois.edu / [Website](http://neerajwagh.com/) / [Twitter](https://twitter.com/neeraj_wagh) / [Google Scholar](https://scholar.google.com/citations?hl=en&user=lCy5VsUAAAAJ)
- John: wei33@illinois.edu
- Yoga: varatha2@illinois.edu / [Website](https://sites.google.com/view/yoga-personal/home) / [Google Scholar](https://scholar.google.com/citations?user=XwL4dBgAAAAJ&hl=en)

### Citation
Wagh, N., Wei, J., Rawal, S., Berry, B., Barnard, L., Brinkmann, B., Worrell, G., Jones, D. &amp; Varatharajah, Y.. (2021). Domain-guided Self-supervision of EEG Data Improves Downstream Classification Performance and Generalizability. _Proceedings of Machine Learning for Health_, in _Proceedings of Machine Learning Research_ 158:130-142 Available from https://proceedings.mlr.press/v158/wagh21a.html.