from create_patches import *

patch_shape = [32, 32, 32]
extraction_shape = [32, 32, 32]

is_lr = False

is_val = False
is_lr_save_patch_name = True
sigma_val = 0.75
simulation_type_dir = 'gauss_sigma_0_75'
image_suffix =  '_lr_zero_pad.nii.gz'

# if is_lr_save_patch_name:
#     if is_val:
#         img_path = 'dataset/hr_data/VSL/val_set/'
#     else:
#         img_path = 'dataset/reg_affine/reg_affine_crop/train_set/'
# else:
#     if is_val:
#         img_path = 'dataset/hr_data/hr_crop/val_set/'
#     else:
#         img_path = 'dataset/hr_data/hr_crop/train_set/'


if is_val:
    img_path = 'dataset/dataset_51/processed_dataset/zero_pad_fu/val_set/'
else:
    img_path = 'dataset/dataset_51/processed_dataset/zero_pad_fu/train_set/'

create_patches(img_path, patch_shape, extraction_shape, is_val, sigma_val, is_lr, simulation_type_dir, image_suffix, is_lr_save_patch_name)
