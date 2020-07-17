import cv2
import numpy as np
from skimage import io
from tqdm import tqdm
from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from os import listdir
from keras.models import load_model
import runway
from runway.data_types import image
import os.path
from os import path


@runway.setup(options={'checkpoint': runway.file(extension='.h5')})
def setup(opts):
  #load_from = "model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep_dsm01proper.h5"
  load_from = opts['checkpoint']
  #if not path.exists(load_from):
  #  print("downloading weights!")
  #  url = "https://www.dropbox.com/s/2260vnfpbqalwgh/model_UNet-Resnet34_DSM_in01_95percOfTrain_8batch_100ep_dsm01proper.h5?dl=1"
  #  import urllib.request
  #  data = urllib.request.urlretrieve(url, load_from)

  model = load_model(load_from)
  return model

@runway.command('classify', inputs={'photo': image}, outputs={'prediction': image})
def classify(model, inputs):

    img = inputs['photo']
    imgnp = np.array(img)
    #print("img:", img, type(img))
    #print("imgnp:", imgnp, type(imgnp))
    images = [imgnp]
    images = np.asarray(images)
    print("loaded shape:", images.shape)

    x_val = images
    ### MODEL ##########################################################################################

    x_val_pred = model.predict(x_val)
    # chop of the last channel which keras needed, for the sake of vis.
    x_val_pred = x_val_pred[:,:,:,0]
    x_val_pred = x_val_pred * 255

    print("pred", x_val_pred.shape)

    out = x_val_pred[0]

    return {'prediction': out}



if __name__ == '__main__':
    runway.run()
