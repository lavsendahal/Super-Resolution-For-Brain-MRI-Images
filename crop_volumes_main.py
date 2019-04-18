import os
import nibabel as nib
import numpy as np

def max_min_norm(image_vol):
    num = image_vol - np.amin(image_vol)
    # print (num.dtype)
    return num / np.amax(num)


def zero_pad_vol(img_path, img_vol_list, is_hr):

    for i in range(0,len(img_vol_list)):
        zero_pad_vol_data = np.zeros((224, 192, 192), dtype='float32')
        img_name = img_path + img_vol_list[i] + '/' + img_vol_list[i] + '_T1_brain.nii.gz'
        print(img_name)
        img = nib.load(img_name)
        affine = img.affine
        img_data = img.get_fdata().transpose(1,0,2)
        print (img_data.shape)
        zero_pad_vol_data[0:218, 0:182, 0:182] = img_data
        zero_pad_vol_data = max_min_norm(zero_pad_vol_data)
        print(np.amax(zero_pad_vol_data))
        print(np.amin(zero_pad_vol_data))

        zero_pad_vol_data = zero_pad_vol_data.transpose(1, 0, 2)
        img_data_save = nib.Nifti1Image(zero_pad_vol_data, affine=affine)

        if is_hr:
            img_name = img_vol_list[i] + '_hr_zero_pad.nii.gz'
            save_dir = 'dataset/dataset_51/processed_dataset/zero_pad_basal/' + img_vol_list[i] + '/'
        else:
            img_name = img_vol_list[i] + '_lr_zero_pad.nii.gz'
            save_dir = 'dataset/dataset_51/processed_dataset/zero_pad_fu/' + img_vol_list[i] + '/'

        nib.save(img_data_save, os.path.join(save_dir, img_name))


def crop_image_vol(img_path, img_vol_list, is_reg):
    for i in range(0, 15):
        img_name = img_path + img_vol_list[i]
        print(img_name)
        img = nib.load(img_name)
        affine = img.affine
        img_data = img.get_fdata()
        img_data = img_data[35:227, 15:143, 0:192]
        img_data = max_min_norm(img_data)
        print(np.amax(img_data))
        print(np.amin(img_data))
        print(img_data.shape)
        print(img_data.dtype)


        img_data_save = nib.Nifti1Image(img_data, affine=affine)

        if is_reg:
            img_name = img_vol_list[i][0:2] + '_reg_affine_crop.nii.gz'
            save_dir = 'temp/reg_affine_crop'
        else:
            img_name = img_vol_list[i][0:2] + '_hr_data_crop.nii.gz'
            save_dir = 'temp/hr_crop'

        nib.save(img_data_save, os.path.join(save_dir, img_name))


is_hr = 0

if is_hr:
    img_vol_type = 'dataset/dataset_51/original_dataset/mni_basal/'
    img_vol_type_list = os.listdir(img_vol_type)
else:
    img_vol_type = 'dataset/dataset_51/original_dataset/mni_fu/'
    img_vol_type_list = os.listdir(img_vol_type)

# crop_image_vol(img_vol_type, img_vol_type_list, is_hr)
zero_pad_vol(img_vol_type, img_vol_type_list, is_hr)
