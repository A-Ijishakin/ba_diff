import nibabel as nib 
import os
import numpy as np 
import torch
import torch.nn as nn
import skimage.io
import matplotlib.animation as animation  
import matplotlib.pyplot as plt
# from PIL import Image
# from torchvision import transforms
from sklearn.metrics import normalized_mutual_info_score
from scipy.stats import pearsonr 
from glob import glob 
from pydicom import dcmread  
from monai.transforms import (
    AddChannel, 
    Resize,
    Spacing,
    ResizeWithPadOrCrop
)   
import ants 
from HD_BET.run import run_hd_bet 
# from config import TrainConfig 
# torch.backends.cudnn.enabled=False 

brain_age = True 


def animate(img, start, slice_range, anim_name, interval, slice_array=(0, slice(None), slice(None), 65)):
  """ animate 
  
  A function which creates a video showing various slices from a brain image. 

  Args:
    - img: The brain image to be animated. 
    - slice_range: The slices to be looped through 
    - anim_name: The name that the animation will be saved under
    - interval: The time interval between frames  

  Output:
    - ani: The animation                                                     """

  if not start:
    start = 0 
  fig, ax = plt.subplots() 

  # ims is a list of lists, each row is a list of artists to draw in the
  # current frame; here we are just animating one artist, the image, in
  # each frame
  ims = [] 
  plt.axis('off')
  for i in range(start, slice_range):
      valid_idx = [idx for idx, val in enumerate(slice_array) if val != slice(None)].pop()
      slice_array[valid_idx] = i 
      ax.set_title(f'Slice {i}') 
      im = ax.imshow(img[slice_array], animated=True, cmap='gray', origin='lower'); 
    
      ims.append([im])

  ani = animation.ArtistAnimation(fig, ims, interval=interval, blit=True,
                                repeat_delay=1000)
  
  writergif = animation.PillowWriter(fps=30)
  ani.save(anim_name, writer=writergif) 

  
def weight_init(model):
  """ weight init  
  
  A function which initialises (PyTorch) model weights in-order to have mean zero and a standard deviation of 0.01. 
  Args: 
    - model: The model which will be operated upon. 
  Output:
    - model: The model following weight initialisation. 
  """
  if type(model) == nn.Linear:
      nn.init.normal_(model.weight, std=0.01)
      
class MutualInformation(nn.Module):
	def __init__(self, sigma=0.1, num_bins=256, normalize=True):
		super(MutualInformation, self).__init__()

		self.sigma = sigma
		self.num_bins = num_bins
		self.normalize = normalize
		self.epsilon = 1e-10

		self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device='cuda').float(), requires_grad=False)


	def marginalPdf(self, values):
		residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
		kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
		
		pdf = torch.mean(kernel_values, dim=1)
		normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
		pdf = pdf / normalization
		
		return pdf, kernel_values


	def jointPdf(self, kernel_values1, kernel_values2):

		joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
		normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
		pdf = joint_kernel_values / normalization

		return pdf


	def getMutualInformation(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W
			return: scalar
		'''

		# Torch tensors for images between (0, 1)
		input1 = input1*255
		input2 = input2*255

		B, C, H, W = input1.shape
		assert((input1.shape == input2.shape))

		x1 = input1.view(B, H*W, C)
		x2 = input2.view(B, H*W, C)
		
		pdf_x1, kernel_values1 = self.marginalPdf(x1)
		pdf_x2, kernel_values2 = self.marginalPdf(x2)
		pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

		H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
		H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
		H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

		mutual_information = H_x1 + H_x2 - H_x1x2
		
		if self.normalize:
			mutual_information = 2*mutual_information/(H_x1+H_x2)

		return mutual_information


	def forward(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W
			return: scalar
		'''
		return self.getMutualInformation(input1, input2)


def dicom_volume(path): 
    
    """ function for getting DICOM volumes and convert to numpy array. """
    
    list_dcm = glob(f'{path}/*')
    # list_index = []
    # for path_dcm in list_dcm:
    #     list_index.append(int(path_dcm.split('/')[-1].replace('Image-', '').replace('.dcm', ''))) # parsing and adding the index of slice
    # list_index.sort() # sort the index
    
    list_arr = [] # array storage
    for dcm in list_dcm:
        ds = dcmread(dcm) # getting slice information via single DICOM file.
        arr = ds.pixel_array # extracting numpy array from DICOM file
        list_arr.append(arr) # stacking to the array storage

    return np.array(list_arr) # return the numpy array 



def normalise(data): 
  """ normalise 
  
  A function which normlaises an image (stored as a numpy.ndarray) to a have values between 0 and 1. 

  Args:
    - data: the image that will be normalised.

  Output:
    - a normalised image. 
  
  """
  return (data - data.min())/(data.max() - data.min())

def to_cpu(tensor):
  return tensor.cpu().detach().numpy()

def load_img(path):
  return nib.load(path).get_fdata() 

def get_reg_accuracy(data_loader, regressor_fn, batch_size):
  """
    compute the accuracy over the supervised training set or the testing set
  """
  predictions, actuals = [], [] 

  # use the appropriate data loader
  for (x, y) in data_loader: 
    x, y = x.to('cuda'), y.to('cuda') 
    
    # use regression function to compute all predictions for each batch 
    predictions.append(regressor_fn(x))
    actuals.append(y.to('cuda')) 
  
  predictions, actuals = [y_hat.detach().cpu().numpy() for y_hat in predictions], [(y.detach().cpu().numpy()) for y in actuals] 
  
  predictions, actuals = [float(y_hat) for array in predictions for y_hat in array], [int(y) for array in actuals for y in array] 
  
  predictions = np.array(predictions).reshape(len(predictions))
  actuals = np.array(actuals).reshape(len(actuals)) 

  # compute the number of accurate predictions
  correlation, p = pearsonr(predictions, actuals) 
  MAE = sum([np.abs(a-b) for a, b in zip(predictions, actuals)])/len(predictions)   

  return correlation, p, MAE  


def normalise(data): 
  """ normalise 
  
  A function which normlaises an image (stored as a numpy.ndarray) to a have values between 0 and 1. 

  Args:
    - data: the image that will be normalised.

  Output:
    - a normalised image. 
  
  """
  return (data - data.min())/(data.max() - data.min())


def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of lengtn n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    return img_ras_axes


def align_volume_to_ref(volume, aff, aff_ref=None, return_aff=False, n_dims=None):
    """This function aligns a volume to a reference orientation (axis and direction) specified by an affine matrix.
    :param volume: a numpy array
    :param aff: affine matrix of the floating volume
    :param aff_ref: (optional) affine matrix of the target orientation. Default is identity matrix.
    :param return_aff: (optional) whether to return the affine matrix of the aligned volume
    :param n_dims: (optional) number of dimensions (excluding channels) of the volume. If not provided, n_dims will be
    inferred from the input volume.
    :return: aligned volume, with corresponding affine matrix if return_aff is True.
    """

    # work on copy
    new_volume = volume.copy()
    aff_flo = aff.copy()

    # default value for aff_ref
    if aff_ref is None:
        aff_ref = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # extract ras axes
    if n_dims is None:
        n_dims, _ = get_dims(new_volume.shape)
    ras_axes_ref = get_ras_axes(aff_ref, n_dims=n_dims)
    ras_axes_flo = get_ras_axes(aff_flo, n_dims=n_dims)

    # align axes
    aff_flo[:, ras_axes_ref] = aff_flo[:, ras_axes_flo]
    for i in range(n_dims):
        if ras_axes_flo[i] != ras_axes_ref[i]:
            new_volume = np.swapaxes(new_volume, ras_axes_flo[i], ras_axes_ref[i])
            swapped_axis_idx = np.where(ras_axes_flo == ras_axes_ref[i])
            ras_axes_flo[swapped_axis_idx], ras_axes_flo[i] = ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]

    # align directions
    dot_products = np.sum(aff_flo[:3, :3] * aff_ref[:3, :3], axis=0)
    for i in range(n_dims):
        if dot_products[i] < 0:
            new_volume = np.flip(new_volume, axis=i)
            aff_flo[:, i] = - aff_flo[:, i]
            aff_flo[:3, 3] = aff_flo[:3, 3] - aff_flo[:3, i] * (new_volume.shape[i] - 1)

    if return_aff:
        return new_volume, aff_flo
    else:
        return new_volume
def reorder_voxels(vox_array, affine, voxel_order):
    '''Reorder the given voxel array and corresponding affine.
    Parameters
    ----------
    vox_array : array
        The array of voxel data
    affine : array
        The affine for mapping voxel indices to Nifti patient space
    voxel_order : str
        A three character code specifing the desired ending point for rows,
        columns, and slices in terms of the orthogonal axes of patient space:
        (l)eft, (r)ight, (a)nterior, (p)osterior, (s)uperior, and (i)nferior.
    Returns
    -------
    out_vox : array
        An updated view of vox_array.
    out_aff : array
        A new array with the updated affine
    reorient_transform : array
        The transform used to update the affine.
    ornt_trans : tuple
        The orientation transform used to update the orientation.
    '''
    #Check if voxel_order is valid
    voxel_order = voxel_order.upper()
    if len(voxel_order) != 3:
        raise ValueError('The voxel_order must contain three characters')
    dcm_axes = ['LR', 'AP', 'SI']
    for char in voxel_order:
        if not char in 'LRAPSI':
            raise ValueError('The characters in voxel_order must be one '
                             'of: L,R,A,P,I,S')
        for idx, axis in enumerate(dcm_axes):
            if char in axis:
                del dcm_axes[idx]
    if len(dcm_axes) != 0:
        raise ValueError('No character in voxel_order corresponding to '
                         'axes: %s' % dcm_axes)

    #Check the vox_array and affine have correct shape/size
    if len(vox_array.shape) < 3:
        raise ValueError('The vox_array must be at least three dimensional')
    if affine.shape != (4, 4):
        raise ValueError('The affine must be 4x4')

    #Pull the current index directions from the affine
    orig_ornt = nib.io_orientation(affine)
    new_ornt = nib.orientations.axcodes2ornt(voxel_order)
    ornt_trans = nib.orientations.ornt_transform(orig_ornt, new_ornt)
    orig_shape = vox_array.shape
    vox_array = nib.apply_orientation(vox_array, ornt_trans)
    aff_trans = nib.orientations.inv_ornt_aff(ornt_trans, orig_shape)
    affine = np.dot(affine, aff_trans)

    return (vox_array, affine, aff_trans, ornt_trans)

def preprocess(input_path, use_gpu=False, save_path=None, skull_strip=False, register=False, project_name=None, return_raw=False):   
	if not os.path.exists('./{}/temp_data'.format(project_name)):
		os.makedirs('./{}/temp_data'.format(project_name))
	reoriented_path = './{}/temp_data/reorient.nii.gz'.format(project_name)
	stripped_path = './{}/temp_data/stripped.nii.gz'.format(project_name)

	orig_nii = nib.load(input_path)
	orig_arr, orig_affine = normalise(np.asarray(orig_nii.dataobj)), orig_nii.affine
	reoriented_arr, reoriented_affine, *_ = reorder_voxels(orig_arr, orig_affine, 'RAS')
	new_image = nib.Nifti1Image(reoriented_arr, reoriented_affine)
	nib.save(new_image, reoriented_path) 
	
	
	run_hd_bet(reoriented_path, stripped_path, mode='fast', device='cuda:0')
	
	try:
		if skull_strip:
			if not os.path.exists('./{}/temp_data'.format(project_name)):
				os.makedirs('./{}/temp_data'.format(project_name))
			reoriented_path = './{}/temp_data/reorient.nii.gz'.format(project_name)
			stripped_path = './{}/temp_data/stripped.nii.gz'.format(project_name)

			orig_nii = nib.load(input_path)
			orig_arr, orig_affine = normalise(np.asarray(orig_nii.dataobj)), orig_nii.affine
			reoriented_arr, reoriented_affine, *_ = reorder_voxels(orig_arr, orig_affine, 'RAS')
			new_image = nib.Nifti1Image(reoriented_arr, reoriented_affine)
			nib.save(new_image, reoriented_path) 
			if use_gpu:
				run_hd_bet(reoriented_path, stripped_path, mode='fast', device='cuda:0')
				# cmd = 'wsl ~ -e -sh -c "hd-bet -i {} -o {} -mode fast"'.format(reoriented_path, stripped_path)
			# else:
			# cmd = 'hd-bet -i {} -o {} -mode fast -device cpu'.format(reoriented_path, stripped_path)       
			# # os.system(cmd) 
			if not os.path.exists(stripped_path): 
				print('skull-stripping failed - skipping this image: {}'.format(input_path))
				return None 
			if not register:
				input_path = stripped_path
			else:
				registered_path = './{}/temp_data/registered.nii.gz'.format(project_name)
				fixed = ants.image_read('Data/MNI152_T1_1mm_brain.nii')
				fixed_nii = nib.load('Data/MNI152_T1_1mm_brain.nii') 
				fixed_arr, fixed_affine = np.asarray(fixed_nii.dataobj), fixed_nii.affine
				moving = ants.n4_bias_field_correction(ants.image_read(stripped_path))
				mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='AffineFast')
				im = mytx['warpedmovout'].numpy()
				new_image = nib.Nifti1Image(im, fixed_affine)
				nib.save(new_image, registered_path)
				input_path = registered_path

		orig_nii = nib.load(input_path)
		orig_arr, orig_affine = np.asarray(orig_nii.dataobj), orig_nii.affine
		reoriented_arr, reoriented_affine, *_ = reorder_voxels(orig_arr, orig_affine, 'RAS')
		reoriented_arr = AddChannel()(reoriented_arr)
		resampled_arr =  Spacing(pixdim=(1.4, 1.4, 1.4), mode='bilinear')(reoriented_arr, reoriented_affine)[0]
		pad_size = 130 
		min_dim = 85
		crop_pad = ResizeWithPadOrCrop(spatial_size=(pad_size,pad_size, pad_size))    
		mask = resampled_arr.squeeze()>resampled_arr.squeeze().std()
		if not (resampled_arr.shape[-1] > min_dim and resampled_arr.shape[-2] > min_dim and resampled_arr.shape[-3] > min_dim):
			return None
		c = 1e9 
		c1 = 1e9
		num_sag_slices = resampled_arr.shape[1]
		for frac in [0.4, 0.45, 0.5, 0.55, 0.6]:
			sl = int(frac*num_sag_slices)          
			z = np.argmax(mask[sl,:,:], axis=1)
			z = np.where(z==0, np.inf, z).min().astype(int)

			z1 = np.argmax(np.fliplr(mask[sl,:,:]), axis=1)
			z1 = np.where(z1==0, np.inf, z1).min().astype(int)

			if z < c:
				c = z
			if z1 < c1:
				c1 = z1
		if c < 0 or c1 < 0:
			print('Cropping failed - skipping this image: {}'.format(input_path))
			return None
		temp_arr = resampled_arr[:,:,:,np.maximum(resampled_arr.shape[-1]-c1-pad_size,c):-c1]

		# print(f'Temp array type is {type(reoriented_arr)}')
		mask = temp_arr.squeeze()>temp_arr.squeeze().std()

		a, b, a1, b1 = 1e9, 1e9, 1e9, 1e9
		num_slices = temp_arr.shape[-1]
		for frac in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]: 
			sl = int(frac*num_slices)

			y = np.argmax(mask[:,:,sl], axis=1)
			y = np.where(y==0, np.inf, y).min().astype(int)
			y1 = np.argmax(np.fliplr(mask[:,:,sl]), axis=1)
			y1 = np.where(y1==0, np.inf, y1).min().astype(int)

			x = np.argmax(mask[:,:,sl], axis=0)
			x = np.where(x==0, np.inf, x).min().astype(int)
			x1 = np.argmax(np.flipud(mask[:,:,sl]), axis=0)
			x1 = np.where(x1==0, np.inf, x1).min().astype(int)

			if x < a:
				a = x
			if y < b:
				b = y
			if x1 < a1:
				a1 = x1
			if y1 < b1:
				b1 = y1
		if a < 0 or a1 < 0 or b < 0 or b1 < 0:
			print('Cropping failed- skipping this image ({})'.format(input_path))
			return None
		processed_arr = crop_pad(temp_arr[:,a:-a1, b:-b1,:]) 
		# print(f'Processed array type is {type(reoriented_arr)}')
		if save_path:
			new_image = nib.Nifti1Image(processed_arr, np.eye(4))
			nib.save(new_image, save_path)


		if return_raw:
			return orig_arr, processed_arr
		else:  
			return processed_arr

	except Exception as e:
		print('***SKIPPING IMAGE {} AS PREPROCESSING FAILED***, see error below: \n\n'.format(input_path))
		print(e)
		return None  




