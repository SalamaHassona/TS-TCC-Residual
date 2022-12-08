# Self-supervised contrastive learning for chaotic time-series classification

## Abstract
We apply a self-supervised contrastive learning approach to reconstruct two-parameter bifurcation diagrams of the chaotic nonlinear dynamical systems. By using only 1% of the dataset labels we can reconstruct the diagrams with the accuracy of about 92%. Furthermore, the method does not require prior knowledge of the system or the labeling of the whole nonlinear time-series dataset, which makes it as useful as other statistical methods, for example, the surrogate data ones, the 0-1 test for chaos, and others. We use the transformed Temporal and Contextual Contrasting (TS-TCC) framework and apply the residual components and scaling as our data augmentation techniques to train the TS-TCC framework. We test our approach against the regular TS-TCC model and the supervised approach, obtaining very promising results.

## Requirmenets:
- Python3.x
- Pytorch==1.7
- Numpy
- Sklearn
- openpyxl (for classification reports)

We used public dataset in this study:
- [Welding Arc](https://drive.google.com/drive/folders/19fb0V4TLiVvetVPA3bNpmBsiqbGQ_cUV?usp=sharing)

### Preparing datasets
The data should be in a separate folder called "data" inside the project folder.
Inside that folder, you should have a separate folders; one for each dataset. Each subfolder should have "train.pt", "val.pt" and "test.pt" files.
The structure of data files should in dictionary form as follows:
`train.pt = {"samples": data, "labels: labels}`, and similarly `val.pt`, and `test.pt`

The details of preprocessing is as follows:
#### 1- Welding Arc dataset L=1:
Create a folder named `data_files` in the path `data_preprocessing/arc/l1`.
Download the dataset files and place them in this folder. 

Run the script `preprocess_arc_l1.py` to generate the pt files.

#### 2- Welding Arc dataset L=1.1:
Create a folder named `data_files` in the path `data_preprocessing/arc/l1_1`.
Download the dataset files and place them in this folder. 

Run the script `preprocess_arc_l1_1.py` to generate the pt files.

#### 3- Welding Arc dataset L=2:
Create a folder named `data_files` in the path `data_preprocessing/arc/l2`.
Download the dataset files and place them in this folder. 

Run the script `preprocess_arc_l2.py` to generate the pt files.

### Configurations
The configuration files in the `config_files` folder should have the same name as the dataset folder name.
Please use bash files to see each command used to run experiments.

## Training TS-TCC 
You can select one of several training modes:
 - Random Initialization (random_init)
 - Supervised training (supervised)
 - Self-supervised training (self_supervised)
 - Fine-tuning the self-supervised model (fine_tune)
 - Training a linear classifier (train_linear)

The code allows also setting a name for the experiment, and a name of separate runs in each experiment.
It also allows the choice of a random seed value.

To use these options:
```
python main.py --experiment_description exp1 --run_description run_1 --seed 123 --training_mode random_init --selected_dataset arc
```
Note that the name of the dataset should be the same name as inside the "data" folder, and the training modes should be
the same as the ones above.

To train the model for the `fine_tune` and `train_linear` modes, you have to run `self_supervised` first.


## Results
- The experiments are saved in "experiments_logs" directory by default (you can change that from args too).
- Each experiment will have a log file and a final classification report in case of modes other that "self-supervised".

## Citation
If you found this work useful for you, please consider citing it.

## Credits
Please note that this repository is a fork of the original paper "Time-Series Representation Learning via Temporal and Contextual Contrasting (TS-TCC)" [[Paper](https://www.ijcai.org/proceedings/2021/0324.pdf)]. We would like to thank the authors for releasing the code that allowed us to use and modify it for the chaotic time-series scenario.

## Contact
For any issues/questions regarding the paper or reproducing the results, please contact me.   
Salama Hassona  
Opole University of Technology, Opole, Opolskie, Poland.    
Email: salama.hassona{at}gmail.com  