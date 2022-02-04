from mrcnn import model as modellib, utils
from fish import FishConfig
import onnx
import tf2onnx

class InterenceConfig(FishConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InterenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config,
        model_dir='./log')
model.load_weights('./mask_rcnn_fish_0500.h5', by_name=True)

model_proto, _ = tf2onnx.convert.from_keras(model.keras_model, opset=11)

onnx.save_model(model_proto, './mask_rcnn_fish_0500.onnx')
