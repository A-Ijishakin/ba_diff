import torch 
import pandas as pd
from glob import glob 
import argparse
import subprocess
from experiment import LitModel 
from templates import Example_Diffusion_Model
import nibabel as nib
import torchvision.transforms as transforms  

#produce an argument parser 
parser = argparse.ArgumentParser()
#specify the data directory 
parser.add_argument("--data_dir", type=str, default="data") 
#specify the slice to be used
parser.add_argument('--slice', type=int)
#specify the extension of the files 
parser.add_argument('--ext', type=str)
#specify the checkpoint 
parser.add_argument('--checkpoint', type=str)


#parse the arguments
args = parser.parse_args() 

#initialise the model configuration 
model_configuation = Example_Diffusion_Model()

#initialise the pytorch lightning model 
DiffAE = LitModel(model_configuation) 

#initialise the weights: in this example we use a dummy checkpoint which DOES NOT exist  
state = torch.load(args.checkpoint, map_location='cuda') 

#load the weights 
DiffAE.load_state_dict(state['state_dict'])

#set into evaluation mode 
DiffAE.eval().to('cuda')

#instatiate the encoder
encoder = DiffAE.model.encoder

#instantiate the network
network = DiffAE.model.network

#memory allocation for predicted age
predicted_age = [] 

#memory allocation for the IDs
IDs = []

#specify the transformations to be used 
transforms_list = [transforms.ToPILImage(), 
                          transforms.Resize((128, 128)), 
                          transforms.ToTensor()]
       
with torch.no_grad():
    for data in glob(f'{args.data_dir}/*{args.ext}'): 
        #load the image and take a slice 
        x = nib.load(data).get_fdata()[:, :, args.slice] 
        #transform the image
        x = transforms_list(x)
        #encode
        z_sem = encoder(x) 
        #predict 
        y_pred = network(z_sem)
        #append the predicted age 
        predicted_age.append(y_pred)

        #append the ID
        IDs.append(data.replace(args.ext, ''))

#convert the results to a dataframe       
results = pd.DataFrame({'ID': IDs, 'Predicted_age': predicted_age}) 
#save the results
results.to_csv('predicted_ages.csv', index=False) 