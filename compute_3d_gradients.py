import numpy as np
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt

def compute_3d_grad(img1, img2, patch_size):

    img1 = tf.reshape(img1, [1, patch_size,patch_size , -1, 1])
    img2 = tf.reshape(img2, [1, patch_size, patch_size,-1, 1])

    sy = tf.constant([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    sx = tf.transpose(sy)

    szx = [ sx, sx, sx]
    szx = tf.expand_dims(szx,axis=-1)
    szx = tf.expand_dims(szx,axis=-1)

    szy = [sy, sy, sy]
    szy = tf.expand_dims(szy, axis=-1)
    szy = tf.expand_dims(szy, axis=-1)

    s_all1 = tf.constant([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    s_all0 = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    s_all_neg1 =  tf.constant([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])

    s_all1 = tf.constant([[1.0, 3.0, 1.0], [3.0, 6.0, 3.0], [1.0, 3.0, 1.0]])
    s_all0 = tf.constant([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    s_all_neg1 =  tf.constant([[-1.0, -3.0, -1.0], [-3.0, -6.0, -3.0], [-1.0, -3.0, -1.0]])

    szz = [s_all1, s_all0, s_all_neg1]
    szz = tf.expand_dims(szz, axis=-1)
    szz = tf.expand_dims(szz, axis=-1)

    Gx1 = tf.nn.conv3d(img1, szx, strides = [1,1,1,1,1], padding = 'VALID')
    Gy1 = tf.nn.conv3d(img1, szy, strides = [1,1,1,1,1], padding ='VALID')
    Gz1 = tf.nn.conv3d(img1, szz, strides = [1,1,1,1,1], padding = 'VALID')

    grdmg1 = (tf.math.square(Gx1) + tf.math.square(Gy1) + tf.math.square(Gz1))


    return grdmg1
    # grdmg1 = tf.math.reduce_sum(tf.math.square(Gx1) + tf.math.square(Gy1) + tf.math.square(Gz1))
    # grdmg1 = tf.math.sqrt(grdmg1)
    #
    # Gx2 = tf.nn.conv3d(img2, szx, strides = [1,1,1,1,1], padding = 'VALID')
    # Gy2 = tf.nn.conv3d(img2, szy, strides = [1,1,1,1,1], padding = 'VALID')
    # Gz2 = tf.nn.conv3d(img2, szz, strides = [1,1,1,1,1], padding = 'VALID')
    #
    # # grdmg2 = tf.math.reduce_sum(tf.math.square(Gx2) + tf.math.square(Gy2) + tf.math.square(Gz2))
    # # grdmg2 = tf.math.sqrt(grdmg2)
    #
    # im1 = tf.math.square(Gx1) + tf.math.square(Gy1) + tf.math.square(Gz1)
    # im2 = tf.math.square(Gx2) + tf.math.square(Gy2) + tf.math.square(Gz2)
    # return tf.math.reduce_sum(im1)
    # # return tf.math.abs((grdmg1- grdmg2))

sess = tf.Session()
reg_path = 'dataset/reg_affine/reg_affine_crop/test_set/01/01_reg_affine_crop.nii.gz'

# hr_path = 'dataset/reg_affine/reg_affine_crop/test_set/01/01_reg_affine_crop.nii.gz'
hr_path = 'dataset/hr_data/hr_crop/test_set/01/01_hr_data_crop.nii.gz'

reg_vol = nib.load(reg_path)

hr_vol = nib.load(hr_path)

reg_vol_data = reg_vol.get_fdata()
hr_vol_data = hr_vol.get_fdata()

print (hr_vol.shape)
reg_vol_crop = reg_vol_data[60:124, 60:124, 60:124]
hr_vol_crop = hr_vol_data[60:124, 60:124, 60:124]

reg_vol_crop_flat = reg_vol_crop.flatten()
hr_vol_crop_flat = hr_vol_crop.flatten()

reg_vol_crop_flat = np.expand_dims(reg_vol_crop_flat, axis=-1)
reg_vol_crop_flat = np.expand_dims(hr_vol_crop_flat, axis=-0)

reg_vol_crop_flat = reg_vol_crop_flat.astype('float32')


hr_vol_crop_flat = np.expand_dims(hr_vol_crop_flat, axis=-1)
hr_vol_crop_flat = np.expand_dims(hr_vol_crop_flat, axis=-0)

hr_vol_crop_flat = hr_vol_crop_flat.astype('float32')

abc= compute_3d_grad(hr_vol_crop_flat, hr_vol_crop_flat, patch_size=64)
result_abc = sess.run(abc)
print (result_abc.shape)
fig, axes = plt.subplots(2,6)


image_slice_hr_sq = np.squeeze(result_abc)
print(image_slice_hr_sq.shape)
axes[0,0].imshow(image_slice_hr_sq[:, :, 5], cmap='gray')
axes[0,1].imshow(image_slice_hr_sq[:, :, 10], cmap='gray')
axes[0,2].imshow(image_slice_hr_sq[:, :, 15], cmap='gray')
axes[0,3].imshow(image_slice_hr_sq[:, :, 20], cmap='gray')
axes[0,4].imshow(image_slice_hr_sq[:, :, 25], cmap='gray')
axes[0,5].imshow(image_slice_hr_sq[:, :, 30], cmap='gray')

#

axes[1,0].imshow(hr_vol_crop[:, :, 5], cmap='gray')
axes[1,1].imshow(hr_vol_crop[:, :, 10], cmap='gray')
axes[1,2].imshow(hr_vol_crop[:, :, 15], cmap='gray')
axes[1,3].imshow(hr_vol_crop[:, :, 20], cmap='gray')
axes[1,4].imshow(hr_vol_crop[:, :, 25], cmap='gray')
axes[1,5].imshow(hr_vol_crop[:, :, 30], cmap='gray')
plt.show(block=True)