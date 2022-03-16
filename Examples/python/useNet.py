import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.python.saved_model import tag_constants
import sys
tf.contrib.resampler
print(sys.version)
print(tf.__version__)


class HFNet:
    def __init__(self, model_path="./Examples/python/model/hfnet"):
        outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
        self.session = tf.Session()
        self.image_ph = tf.placeholder(tf.float32, shape=(None, None, 3))

        net_input = tf.image.rgb_to_grayscale(self.image_ph[None])
        tf.saved_model.loader.load(
            self.session, [tag_constants.SERVING], str(model_path),
            clear_devices=True,
            input_map={'image:0': net_input})

        graph = tf.get_default_graph()
        self.outputs = {n: graph.get_tensor_by_name(
            n + ':0')[0] for n in outputs}
        self.nms_radius_op = graph.get_tensor_by_name(
            'pred/simple_nms/radius:0')
        self.num_keypoints_op = graph.get_tensor_by_name(
            'pred/top_k_keypoints/k:0')
        print("python初始化完成")

    def inference(self, image_name="test.jpg"):
        nms_radius = 4
        num_keypoints = 50
        print("提取", image_name, "特征")
        image = cv2.imread(image_name)
        #cv2.imshow(image_name, image)
        # cv2.waitKey(3)
        inputs = {
            self.image_ph: image[..., ::-1].astype(np.float),
            self.nms_radius_op: nms_radius,
            self.num_keypoints_op: num_keypoints,
        }
        query = self.session.run(self.outputs, feed_dict=inputs)
        localDes = np.asarray(query['local_descriptors'])
        globalDes = np.asarray(query['global_descriptor'])
        keypoints = np.asarray(query['keypoints'])
        return globalDes, keypoints, localDes


if __name__ == "__main__":
    model_path = "./Examples/python/model/hfnet"
    outputs = ['global_descriptor', 'keypoints', 'local_descriptors']
    hfnet = HFNet(model_path)
    print("hfnet")
    image_name = "Data/db1.jpg"
    globalDes, keypoints, localDes = hfnet.inference(image_name)
    #for i in globalDes:
    #    print(i)
    #for i in keypoints:
    #    print(i)
    #for i in localDes:
    #    print(i)
