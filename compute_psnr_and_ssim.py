import os
import nibabel as nib
from psnr_and_ssim_utilities import *


simulated_path = 'outputs/prediction/gauss_sigma_1/test_set'
image_path = 'dataset/hr_data/hr_crop/test_set'
simulated_img_vol_list = os.listdir(simulated_path)
img_vol_list = os.listdir(image_path)
print(img_vol_list)
sim_image_suffix = '_pred_reg.nii.gz'

image_suffix = '_hr_data_crop.nii.gz'

psnr_array = np.zeros((len(img_vol_list),1))
ssim_array = np.zeros((len(img_vol_list),1))

psnr_total = 0
ssim_total = 0

for i in range(0, len(img_vol_list)):
    sess = tf.Session()
    simulated_img_name = simulated_img_vol_list[i] + sim_image_suffix
    img_name = img_vol_list[i] + image_suffix
    simulated_img_file_path = simulated_path  +'/' + simulated_img_vol_list[i] + '/' + simulated_img_name
    lr_img_file_path = image_path + '/' + img_vol_list[i] + '/' + img_name

    print(lr_img_file_path, simulated_img_file_path)

    img1 = nib.load(simulated_img_file_path)
    img_data1 = img1.get_fdata()
    # img_data1 *= 1.0 / np.max(img_data1)
    img_data1 = np.expand_dims(img_data1, axis=-1)
    img_data1 = np.expand_dims(img_data1, axis=-0)
    # print(img_data1.shape)
    img_data1 = img_data1.astype('float32')
    # print(img_data1.dtype)

    img2 = nib.load(lr_img_file_path)
    img_data2 = img2.get_fdata()
    img_data2 *= 1.0 / np.max(img_data2)
    img_data2 = np.expand_dims(img_data2, axis=-1)
    img_data2 = np.expand_dims(img_data2, axis=-0)
    # print(img_data2.shape)
    img_data2 = img_data2.astype('float32')
    # print(img_data2.dtype)
    psnr_value = psnr(img_data1, img_data2)
    result_psnr = sess.run(psnr_value)
    psnr_array[i] = result_psnr
    # print('psnr value is:')
    # print(result_psnr)
    psnr_total += result_psnr
    # print('ssim is')
    ssim_value = ssim(img_data1, img_data2)
    result_ssim = sess.run(ssim_value)
    ssim_array[i] = result_ssim
    # print(result_ssim)
    ssim_total += result_ssim


print(psnr_total/len(img_vol_list))
print(ssim_total/len(img_vol_list))

print('all values for psnr are')
print(psnr_array)
print('all values for ssim')
print(ssim_array)