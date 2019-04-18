from sklearn.feature_extraction.image import extract_patches
import numpy as np
import nibabel as nib
import os

def max_min_norm(image_vol):
    num = image_vol - np.amin(image_vol)
    return num / np.amax(num)


def expand_dims_data(image_vol):
    image_vol = np.expand_dims(image_vol, axis=0)
    return image_vol


def getPatches(volFolderName, volPath, patch_size, extraction_reconstruct_step, image_suffix):
    ImgPatches = np.empty(shape=[0, patch_size[0], patch_size[1], patch_size[2]], dtype='float32')
    volFolderPath = os.listdir(volPath)
    img_name = volFolderName + image_suffix
    img_name = os.path.join(volPath, volFolderName, img_name)
    print(img_name)
    img = nib.load(img_name)
    affine = img.affine
    img_data = img.get_fdata().transpose(1,0,2)
    # img_data = np.squeeze(img_data)
    print(img_data.shape)
    ImgPatches_temp = np.empty(shape=[0, patch_size[0], patch_size[1], patch_size[2]], dtype='float32')

    imgs_patches, rows, cols, depths = getGoodPatches(img_data, patch_size, extraction_reconstruct_step)

    ImgPatches_temp = np.append(ImgPatches_temp, imgs_patches, axis=0)
    ImgPatches = np.append(ImgPatches, ImgPatches_temp, axis=0)
    ImgPatches = ImgPatches.astype('float32')
    ImgPatches = np.expand_dims(ImgPatches, axis=4)


    # return img_data, affine
    return ImgPatches, rows, cols, depths, affine


def getGoodPatches(img_data, patch_size, extraction_reconstruct_step):
    patch_shape = (patch_size[0], patch_size[1], patch_size[2])
    extraction_step = (extraction_reconstruct_step[0], extraction_reconstruct_step[1], extraction_reconstruct_step[2])
    img_patches = extract_patches(img_data, patch_shape, extraction_step)
    rows = []
    cols = []
    depths = []
    for i in range(0, img_patches.shape[0]):
        for j in range(0, img_patches.shape[1]):
            for k in range(0, img_patches.shape[2]):
                cubeSum = np.sum(img_patches[i, j, k, patch_size[0] - 2:patch_size[0] + 2, patch_size[1] - 2:patch_size[1] + 2,
                                 patch_size[2] - 2:patch_size[2] + 2])
                if cubeSum >= 0:
                    rows.append(i)
                    cols.append(j)
                    depths.append(k)
    selected_img_patches = img_patches[rows, cols, depths, :, :, :]
    lower_res_all_patches = np.empty(shape=[0, patch_size[0], patch_size[1], patch_size[2]], dtype='float32')
    for i in range(selected_img_patches.shape[0]):
        lower_res_patches = expand_dims_data(selected_img_patches[i])
        lower_res_all_patches = np.append(lower_res_all_patches, lower_res_patches, axis=0)

    return lower_res_all_patches, rows, cols, depths


def getFinalPrediction(patchesPredicted, rows, cols, depths, patch_size, extraction_reconstruct_step, image_dim):
    image_rows = image_dim[0]
    image_cols = image_dim[1]
    image_depth = image_dim[2]
    labelOriginal = np.zeros((image_rows, image_cols, image_depth))
    labelOriginalall = labelOriginal

    for index in range(0, len(rows)):
        print('Processing patches: ', index + 1, '/', len(rows))
        row = rows[index]
        col = cols[index]
        dep = depths[index]
        start_row = row * extraction_reconstruct_step[0]
        start_col = col * extraction_reconstruct_step[1]
        start_dep = dep * extraction_reconstruct_step[2]
        #         print (start_row,start_col,start_dep)
        patch_volume = patchesPredicted[index, :, :, :]
        for i in range(0, patch_size[0]):
            for j in range(0, patch_size[1]):
                for k in range(0, patch_size[2]):
                    label0probtemp = patch_volume[i][j][k]
                    labelOriginalall[start_row + i][start_col + j][start_dep + k] = label0probtemp
    #                     print (start_row+i, start_col+j, start_dep+k)
    return labelOriginalall


def writePrediction(volFolderName, model, volPath, patch_size, extraction_reconstruct_step, checkpoint_filename, pred_folder, image_dim, image_suffix):
    checkpoint_filepath = 'output_server/' + pred_folder + checkpoint_filename
    model.load_weights(checkpoint_filepath, reshape=True)

    SegmentedVolume = np.zeros((image_dim[0], image_dim[1], image_dim[2]))

    ImgPatches, rows, cols, depths, affine = getPatches(volFolderName, volPath, patch_size, extraction_reconstruct_step, image_suffix)
    # ImgPatches ,affine = getPatches(volFolderName, volPath, patch_size, extraction_reconstruct_step,
    #                                                     image_suffix)
    # ImgPatches = np.expand_dims(ImgPatches,axis=0)
    # ImgPatches = np.expand_dims(ImgPatches,axis=-1)
    patchesPredicted = model.predict(ImgPatches, batch_size=1)

    print('shape of patches predicted')
    print(patchesPredicted.shape)
    patchesPredicted = np.squeeze(patchesPredicted)

    finalPrediction = getFinalPrediction(patchesPredicted, rows, cols, depths, patch_size, extraction_reconstruct_step, image_dim)
    print('The segmentation for this volume is complete ! ')
    data = finalPrediction.transpose(1,0,2)
    # img = nib.Nifti1Image(patchesPredicted, affine=affine)
    img = nib.Nifti1Image(data, affine=affine)
    img_name = volFolderName + '_pred_fu.nii.gz'
    predictSaveDir = 'prediction'
    nib.save(img, os.path.join('outputs', predictSaveDir, img_name))
    print('-' * 50)

