import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import Dataloader
from tqdm import tqdm

framObjTrain = {'img': [],
                'mask': [],
                }


def predict(valMap, model, num=None):
    ## getting and proccessing val data
    if num:
        img = valMap['img'][0:num]
        mask = valMap['mask'][0:num]
    else:
        img = valMap['img']
        mask = valMap['mask']

    imgProc = []
    for image in img:
        img_resized = cv2.resize(image, (320, 240))
        imgProc.append(img_resized)

    imgProc = np.array(imgProc)

    predictions = model.predict(imgProc)
    predictions = np.array(predictions)

    resized_imgProc = []
    for img in imgProc:
        resized_img = cv2.resize(img, (640, 480))
        resized_imgProc.append(resized_img)

    resized_predictions = []
    for pred_imgs in predictions:
        resized_imgs = []
        for pred_img in pred_imgs:
            resized_img = cv2.resize(pred_img, (640, 480))
            resized_imgs.append(resized_img)
        resized_predictions.append(resized_imgs)

    resized_predictions = np.array(resized_predictions)

    return resized_predictions[4], imgProc, mask


def Plotter(img, predMask, groundTruth, result_dir):
    plt.figure(figsize=(9, 3))

    img = img.astype('uint8')
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display

    plt.title('Input image')
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(predMask, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display
    plt.title('Predicted Result')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(groundTruth, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display
    plt.title('Actual Label Mask')
    plt.savefig(os.path.join(result_dir, f"combined_result_{i + 1}.png"))
    # plt.show()


if __name__ == '__main__':
    print("Load Data...")
    root_path = r'../Dataset/test/'
    img_idx, anno_idx = Dataloader.gerAddress(root_path)
    frameObjTrain = Dataloader.LoadData(img_idx=img_idx, anno_idx=anno_idx)
    print("Load Model...")
    unet = tf.keras.models.load_model('./ResUnet.h5')
    print(f"Number of loaded samples: {len(frameObjTrain['img'])}")  # Add this line for debugging
    print(f"Number of loaded ground truth: {len(frameObjTrain['mask'])}")
    total = len(frameObjTrain['img'])
    assert total != 0, 'len(frameObjTrain[\'img\'] = 0)'
    print("Start Predict")

    sixteenPrediction, actuals, masks = predict(frameObjTrain, unet)
    rootDIR = "./output/"
    pred_path = os.path.join(rootDIR, 'pred/')
    mask_path = os.path.join(rootDIR, 'mask/')
    img_path = os.path.join(rootDIR, 'img/')
    combine_dir = os.path.join(rootDIR, 'combine_result/')
    # Make folder
    if os.path.exists(pred_path) is False:
        os.makedirs(pred_path)
    if os.path.exists(mask_path) is False:
        os.makedirs(mask_path)
    if os.path.exists(img_path) is False:
        os.makedirs(img_path)
    if os.path.exists(combine_dir) is False:
        os.makedirs(combine_dir)
    # write image to folder
    for i in tqdm(range(total)):
        Plotter(actuals[i], sixteenPrediction[i], masks[i] * 255, combine_dir)
        cv2.imwrite(os.path.join(img_path, f'{i}.png'), actuals[i])
        cv2.imwrite(os.path.join(pred_path, f'{i}.png'), sixteenPrediction[i] * 255)
        cv2.imwrite(os.path.join(mask_path, f'{i}.png'), masks[i] * 255)  # prediction

    print("Prediction completed.")
