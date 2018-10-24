import logging
import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from scipy.spatial.ckdtree import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io

# Extension of feature files.
_DELF_EXT = '.delf'

_CONFIG_PATH = 'delf_config_example.pbtxt'

_DISTANCE_THRESHOLD = 0.8

Feature = namedtuple('Feature', ['locations', 'descriptors'])


def match_features(needle, haystack):
    res = {}
    for filename, feature in haystack.items():
        matched = match(needle.locations, needle.descriptors, feature.locations, feature.descriptors)
        res[filename] = int(matched)
    return res


def match(locations_1, descriptors_1, locations_2, descriptors_2):
    num_features_1 = locations_1.shape[0]
    num_features_2 = locations_2.shape[0]

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    # Perform geometric verification using RANSAC.
    _, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000)

    return sum(inliers)


def load_features(features_dir):
    features = {}
    for feature_file in features_dir.iterdir():
        if feature_file.name.endswith(_DELF_EXT):
            locations, _, descriptors, _, _ = feature_io.ReadFromFile(str(feature_file))
            features[feature_file.name] = Feature(locations, descriptors)

    logging.info("%d feature file loaded" % len(features))

    return features


def extract(filename):
    tf.logging.set_verbosity(tf.logging.INFO)

    image_paths = [filename]

    # Parse DelfConfig proto.
    config = delf_config_pb2.DelfConfig()
    with tf.gfile.FastGFile(_CONFIG_PATH, 'r') as f:
        text_format.Merge(f.read(), config)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Reading list of images.
        filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
        reader = tf.WholeFileReader()
        _, value = reader.read(filename_queue)
        image_tf = tf.image.decode_jpeg(value, channels=3)

        with tf.Session() as sess:
            # Initialize variables.
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            # Loading model that will be used.
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                       config.model_path)
            graph = tf.get_default_graph()
            input_image = graph.get_tensor_by_name('input_image:0')
            input_score_threshold = graph.get_tensor_by_name('input_abs_thres:0')
            input_image_scales = graph.get_tensor_by_name('input_scales:0')
            input_max_feature_num = graph.get_tensor_by_name(
                'input_max_feature_num:0')
            boxes = graph.get_tensor_by_name('boxes:0')
            raw_descriptors = graph.get_tensor_by_name('features:0')
            feature_scales = graph.get_tensor_by_name('scales:0')
            attention_with_extra_dim = graph.get_tensor_by_name('scores:0')
            attention = tf.reshape(attention_with_extra_dim,
                                   [tf.shape(attention_with_extra_dim)[0]])

            locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
                boxes, raw_descriptors, config)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            start = time.clock()

            # # Get next image.
            im = sess.run(image_tf)

            # Extract and save features.
            (locations_out, descriptors_out, feature_scales_out,
             attention_out) = sess.run(
                [locations, descriptors, feature_scales, attention],
                feed_dict={
                    input_image:
                        im,
                    input_score_threshold:
                        config.delf_local_config.score_threshold,
                    input_image_scales:
                        list(config.image_scales),
                    input_max_feature_num:
                        config.delf_local_config.max_feature_num
                })

            # Finalize enqueue threads.
            coord.request_stop()
            coord.join(threads)

            return locations_out, descriptors_out
