import numpy as np
import matplotlib.pyplot as plt

hr_patch_train = np.load('patches_3D/dataset_51/val_img_hr32.npy')
lr_patch_train = np.load('patches_3D/dataset_51/val_img_lr32.npy')
print(hr_patch_train.shape, hr_patch_train.dtype)
print(lr_patch_train.shape, lr_patch_train.dtype)
print(np.max(hr_patch_train))
print(np.max(lr_patch_train))

fig, axes = plt.subplots(2,6)
image_slice_hr = hr_patch_train[90, :, :, :, :]
image_slice_hr_sq = np.squeeze(image_slice_hr)
print(image_slice_hr_sq.shape)
axes[0,0].imshow(image_slice_hr_sq[:, :, 5], cmap='gray')
axes[0,1].imshow(image_slice_hr_sq[:, :, 10], cmap='gray')
axes[0,2].imshow(image_slice_hr_sq[:, :, 15], cmap='gray')
axes[0,3].imshow(image_slice_hr_sq[:, :, 20], cmap='gray')
axes[0,4].imshow(image_slice_hr_sq[:, :, 25], cmap='gray')
axes[0,5].imshow(image_slice_hr_sq[:, :, 30], cmap='gray')
# fig.add_subplot(1, 2, 1)
# plt.imshow(image_slice_hr_sq[:, :, 30], cmap='gray')

image_slice_lr = lr_patch_train[90, :, :, :, :]
image_slice_lr_sq = np.squeeze(image_slice_lr)
print(image_slice_lr_sq.shape)
axes[1,0].imshow(image_slice_lr_sq[:, :, 5], cmap='gray')
axes[1,1].imshow(image_slice_lr_sq[:, :, 10], cmap='gray')
axes[1,2].imshow(image_slice_lr_sq[:, :, 15], cmap='gray')
axes[1,3].imshow(image_slice_lr_sq[:, :, 20], cmap='gray')
axes[1,4].imshow(image_slice_lr_sq[:, :, 25], cmap='gray')
axes[1,5].imshow(image_slice_lr_sq[:, :, 30], cmap='gray')
fig.suptitle('HR and LR Image Slices', fontsize=16)
plt.show(block=True)
# fig.add_subplot(1, 2, 2)
# plt.imshow(image_slice_lr_sq[:, :, 30], cmap='gray')
# plt.show(block=True)
