from sklearn.feature_extraction.image import extract_patches
from scipy.ndimage.filters import gaussian_filter

import os
import time
import numpy as np
import nibabel as nib


def max_min_norm(image_vol):
    num = image_vol - np.amin(image_vol)
    return num / np.amax(num)


def get_lr_patches(img_data, sigma_val, affine, folder_name, simulation_type_dir):
    lr_data = gaussian_filter(img_data, sigma=sigma_val)
    # lr_data *= 1.0 / np.max(lr_data)
    lr_data = max_min_norm(lr_data)
    img = nib.Nifti1Image(lr_data, affine=affine)
    img_name = folder_name + '_gauss_sigma' + '_0_75' + '.nii.gz'
    nib.save(img, os.path.join('outputs', 'simulated_lr', simulation_type_dir, img_name))
    return lr_data


def create_patches(img_path, patch_shape, extraction_shape, is_val, sigma_val, is_lr, simulation_type_dir, image_suffix, is_lr_save_patch_name):
    img_dir = os.listdir(img_path)
    patches_all = np.empty(shape=[0, patch_shape[0], patch_shape[1], patch_shape[2]], dtype='float32')

    for folder_name in img_dir:
        start = time.time()
        name_img = folder_name + image_suffix
        path_img = os.path.join(img_path, folder_name, name_img)
        print('Currently Processing')
        print(path_img)
        print('-' * 60)
        img = nib.load(path_img)
        affine = img.affine
        img_data = img.get_fdata()
        # img_data = img_data[35:227, 15:143, 0:192]

        if is_lr:
            img_data = get_lr_patches(img_data, sigma_val, affine, folder_name,  simulation_type_dir)
        else:
            # img_data = max_min_norm(img_data)
            print (np.amax(img_data))
            print (np.amin(img_data))

        print(img_data.shape)

        patches_extracted = extract_patches(img_data, patch_shape=(patch_shape[0], patch_shape[1], patch_shape[2]), extraction_step=(extraction_shape[0], extraction_shape[1], extraction_shape[2]))
        x = []
        y = []
        z = []

        for i in range(0, patches_extracted.shape[0]):
            for j in range(0, patches_extracted.shape[1]):
                for k in range(0, patches_extracted.shape[2]):
                    cube_sum = np.sum(patches_extracted[i, j, k, :, :, :])
                    if cube_sum >=0:
                        x.append(i)
                        y.append(j)
                        z.append(k)

        all_img_patches = patches_extracted[x, y, z, :, :, :]
        print(all_img_patches.shape)
        patches_all = np.append(patches_all, all_img_patches, axis=0)

        end = time.time()
        print("Elapsed time in seconds: %g" % (end - start))

    patches_all = np.expand_dims(patches_all, axis=4)

    print(patches_all.shape)
    patches_all = patches_all.astype('float32')
    print(patches_all.dtype)
    if is_val:
       if is_lr_save_patch_name:
            save_name = 'patches_3d/dataset_51/' + 'val_img_lr32.npy'
            np.save(save_name, patches_all)
       else:
            save_name = 'patches_3d/dataset_51/' + 'val_img_hr32.npy'
            np.save(save_name, patches_all)

    else:
        if is_lr_save_patch_name:
            save_name = 'patches_3d/dataset_51/' + 'train_img_lr32.npy'
            np.save(save_name, patches_all)
        else:
            save_name = 'patches_3d/dataset_51/' + 'train_img_hr32.npy'
            np.save(save_name, patches_all)
    del patches_all
