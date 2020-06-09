'''
The detection code is partially derived and modified from app-2Class.py by Chengsheng Wang.


'''

from flask import Flask, request, render_template, redirect

import cv2
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
from datetime import timedelta
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)  # avoid caching, which prevent showing the detection/splash result

import os
import sys
import random


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
CKPT_DIR = '/Users/hailieboomboom/Documents/GitHub/models/research/object_detection/data/faster_RCNN_melonstrawberry/frozen_inference_graph.pb'
LABEL_DIR = '/Users/hailieboomboom/Documents/GitHub/models/research/object_detection/data/faster_RCNN_melonstrawberry/fruit_labelmap.pbtxt'

UPLOAD_FOLDER = '/Users/hailieboomboom/Documents/GitHub/models/research/object_detection/image_uploaded'
ALLOWED_EXTENSIONS = set(['jpg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = CKPT_DIR
        self.PATH_TO_LABELS = LABEL_DIR
        self.NUM_CLASSES = 2
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    # load the pre-trained model via the frozen inference graph
    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    # load the label map so that we know what object has been detected
    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image):
        count_result = 0
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                count_result = len(scores)

        # cv2.namedWindow("detection", cv2.WINDOW_NORMAL)

        cv2.imwrite('/Users/hailieboomboom/Documents/GitHub/models/research/object_detection/static/result.jpg',image)

        cv2.waitKey(0)
        return count_result

################################################################
def run_detection():
    user_file_names = next(os.walk(UPLOAD_FOLDER))[2]
    names_chosen = random.choice(user_file_names)
    image = cv2.imread(os.path.join(UPLOAD_FOLDER, names_chosen))
    print('\n-----------------', len([image]), '---------------\n')

    detecotr = TOD()
    detecotr.detect(image)


    print("detection done")

def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath



@app.route('/')
def index():
    return render_template('hello.html')

@app.route('/upload', methods = ['POST'])
def upload():

    if request.method == 'POST' and request.files['image']:
        print("enter!!!!!!!!!!!!!!")
        #app.logger.info(app.config['UPLOAD_FOLDER'])
        img = request.files['image']
        #img_string = request.form['image']
        img_name = secure_filename(img.filename)
        print(img_name)
        #print(img_string)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        print("create upload dir success")
        app.logger.info("saving {}".format(saved_path))
        img.save(saved_path)


        image = cv2.imread(saved_path)
        print("image read successfully")
        detector = TOD()
        count_result = detector.detect(image)
        print("Counting result is ")
        print(count_result)
        return render_template('complete.html', count_result = count_result)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

'''
Main function to run Flask server
'''
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)
