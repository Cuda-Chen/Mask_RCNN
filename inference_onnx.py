import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from mrcnn import model as modellib
from fish import FishConfig, color_splash
import onnxruntime
import skimage
import numpy as np
import datetime

class_names = ['BG', 'fish']

class InterenceConfig(FishConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InterenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
        model_dir='./log')
model.load_weights('./mask_rcnn_fish_0500.h5', by_name=True)

def generate_image(images, molded_images, windows, results):
    results_final = []
    for i, image in enumerate(images):
        final_rois, final_class_ids, final_scores, final_masks = \
            model.unmold_detections(results[0][i], results[3][i], # detections[i], mrcnn_mask[i]
                                   image.shape, molded_images[i].shape,
                                   windows[i])
        results_final.append({
            "rois": final_rois,
            "class_ids": final_class_ids,
            "scores": final_scores,
            "masks": final_masks,
        })
        r = results_final[i]
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                            class_names, r['scores'])
    return results_final

image_file_name = './samples/fish/七星斑.jpg'
image = skimage.io.imread(image_file_name)
images = [image]

sess = onnxruntime.InferenceSession('./mask_rcnn_fish_0500.onnx')

# preprocessing
molded_images, image_metas, windows = model.mold_inputs(images)
anchors = model.get_anchors(molded_images[0].shape)
anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

results = \
    sess.run(None, {"input_image": molded_images.astype(np.float32),
    "input_anchors": anchors,
    "input_image_meta": image_metas.astype(np.float32)})

# postprocessing
results_final = generate_image(images, molded_images, windows, results)

# Splash!
splash = color_splash(image, results_final[0]['masks'])
file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
skimage.io.imsave(file_name, splash)
