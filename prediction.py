from utilities import writePrediction
import os
import time
from model import *

patch_size = [224, 192, 96]
extraction_reconstruct_step = [224, 192, 96]
image_dim = [224, 192, 192]

predictSaveDir = 'prediction'
image_suffix = '_lr_zero_pad.nii.gz'

# if 'prediction' not in os.listdir(os.curdir):
#     os.mkdir('prediction')

VolPath = 'dataset/dataset_51/processed_dataset/zero_pad_fu/test_set/'
checkpoint_filename = 'model-057.h5'

volFolderPath = os.listdir(VolPath)
trained_model_dir = '/outputs_dataset_51/output_loss_fun_mae_and_gd/'

for volFolderName in volFolderPath:
    start = time.time()
    img_name = volFolderName + '_pred_fu.nii.gz'
    print('Path: ', os.path.join('../outputs', predictSaveDir, img_name))
    print('*' * 50)
    model = DenseNet(patch_size=patch_size, growth_rate=8, no_layers=8, loss = custom_loss)
    model.summary()
    writePrediction(volFolderName, model, VolPath, patch_size, extraction_reconstruct_step, checkpoint_filename, trained_model_dir, image_dim, image_suffix)
    end = time.time()
    elapsed_time = end - start
    print('Elapsed time is: ', elapsed_time)