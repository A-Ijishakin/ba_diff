# Semi-Supervised Diffusion Model for Brain Age Prediction

This repository contains the code required to train a Semi-Supervised Diffusion Model for Brain Age Prediction It is an adaptation of the original Diffusion Autoencoder repo found at: 

[[Diffusion Autoencoder Repo](https://diff-ae.github.io/)]

Which was introduced in the paper: 

**Diffusion Autoencoders: Toward a Meaningful and Decodable Representation** \
K. Preechakul, N. Chatthee, S. Wizadwongsa, S. Suwajanakorn 
2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).


The data included in the Workshop paper is not used in this Repo because although alot of the data is available from public datasets, individuals must apply to the relevant consortiums in order to have access. We have provided much detail concerning the datasets in the Appendix, such that it can be requested if one wishees. 

---------------- 
### Training a model 
1. Clone the repo by running: 
    ```
    git clone https://github.com/A-Ijishakin/Contrast-DiffAE.git
    ```
2. Make a virtual environment either natively in python by running:
    ```
    pip install virtualenv 
    ```
    ```
    virtualenv ss_diffage_env
    ``` 
    Or in conda by running:
    ```
    conda create -n _env
    ```
3. Activate that environment
    Native Python:
    ```
    source ./ss_diffage_env/bin/activate 
    ```
    Conda:
    ```
    conda activate ss_diffage_env
    ```

4. Install all of the neccessary dependencies by running: 
    ```
    pip install -r requirement.txt
    ``` 
6. Next ammend the file dataset.py such that it loads in your data accordingly. An example dataloader can be found in the file.     
5. Then config.py must should be ammended such that the hyperparameters used meet your specifications. These arguments exist on the TrainConfig dataclass which starts on line 25. An argument which are particularly of note is:  

-  load_in : This specifies how long training should happen before the age prediction kicks in.
-  batch_size : The size of batches
 
The make_dataset method on the TrainConfig class should also be ammended to load your dataset accordingly. Again examples have been left here as a guide.

1. Following this templates.py needs to be modified according to your model, and data specificiation. Changes to the conf.net_ch_mult, will make your model smaller of bigger for example.
 
2.  Then train.py needs to be ammended such that it calls on the configuration for your dataset/particular model. An example has been left there as well.  

After following the above steps, the model will be ready to train with your specifications and dataset. It is advised that you also inspect the expeiriment.py file as this is the location of the pytorch_lightning class, LitModel, which further defines the training specifications. Methods on this class which should particularly be inspected are: 
- training_step : (line 420) modifications should be made to ensure that the data is loaded in each step appropriately. 
- training_epoch_end : (line 243) modifications should be made to log metrics at the end of each epoch. 
- ModelCheckpoint : (line 1008) modifications should be made to configure checkpointing according to your needs. 

The trainer also includes logging of images and the MSE loss as well, so use of the tensorboard is advised. This can be done by running the following command in a terminal with the aformentioned environment active:
        ```
        tensorboard --logdir=checkpoints
        ``` 
This should open up the tensorboard in a localhost. 

----------------

### Prediction
If you have downloaded the relevant datasets and which to make predictions yourself then you can run the following script. This will work provided that you follow the pre-processing steps outlined in the Appendix of the paper, and have saved the files as niftis. 

1. Run the prediction script:
    ```
    python3 predict.py --data_dir <directory with niftis> --slice <the slice number that you will predict on> --ext <file extension e.g: .nii or .nii.gz> --checkpoint <path to model checkpoint>
    ``` 

The above script will load model weights and save the predictions to a CSV called: predicted_ages.csv. This file will have two columns: ID and predicted age. 