from create_patches import *

patch_shape = [32, 32, 32]
extraction_shape = [32, 32, 32]

is_val = True
is_lr = False
sigma_val = 1
simulation_type_dir = 'gauss_sigma1'
image_suffix =  '_hr_stripped.nii.gz'
is_lr_save_patch_name = True

if is_val:
    img_hr_path = 'processed_data/hr_stripped/val_set/'
else:
    img_hr_path = 'processed_data/hr_stripped/train_set/'

create_patches(img_hr_path, patch_shape, extraction_shape, is_val, sigma_val, is_lr, simulation_type_dir, image_suffix, is_lr_save_patch_name)
