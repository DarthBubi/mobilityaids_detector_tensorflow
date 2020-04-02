import tensorflow as tf
import io
import yaml

from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('mobilityaids_dir', '', 'Path to mobilityadis dataset')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

class_mapping = {
    'person': 0,
    'crutches': 1,
    'walking_frame': 2,
    'wheelchair': 3,
    'push_wheelchair': 4
}

def create_tf_example(image, annotations, image_dir):
    # TODO(user): Populate the following variables from your example.
    height = annotations['height']  # Image height
    width = annotations['width']  # Image width
    filename = annotations['filname']  # Filename of the image. Empty if image is not from file

    path = os.path.joint(image_dir, filename)
    with tf.gfile.GFile(path, 'rb') as fd:
        encoded_image_data = fd.read()
    image_format = b'png'  # b'jpeg' or b'png'
    
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
                # (1 per box)
    # List of normalized top y coordinates in bounding box (1 per box)
    ymins = []
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)

    for bbox in annotations['object']:
        xmins.append(bbox['xmin'])
        xmaxs.append(bbox['xmax'])
        ymins.append(bbox['ymin'])
        ymaxs.append(bbox['ymax'])
        classes_text.append(bbox['name'])

    classes = annotations['classes']  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def _create_tf_record_from_mobilityaids_annotations(
    annotations_file, image_dir ) :
    pass

def main(_):
    imageset_files = [os.path.join(mobilityaids_dir, "ImageSets/TrainSet_DepthJet.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet1.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq1.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq2.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq3.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq4.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TrainSet_RGB.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet1.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq1.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq2.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq3.txt"),
                  os.path.join(mobilityaids_dir, "ImageSets/TestSet2_seq4.txt")]
    
    annotation_dirs = [os.path.join(mobilityaids_dir, "Annotations_DepthJet/"),
                    os.path.join(mobilityaids_dir, "Annotations_DepthJet/"),
                    os.path.join(mobilityaids_dir, "Annotations_DepthJet_TestSet2/"),
                    os.path.join(mobilityaids_dir, "Annotations_DepthJet_TestSet2/"),
                    os.path.join(mobilityaids_dir, "Annotations_DepthJet_TestSet2/"),
                    os.path.join(mobilityaids_dir, "Annotations_DepthJet_TestSet2/"),
                    os.path.join(mobilityaids_dir, "Annotations_RGB/"),
                    os.path.join(mobilityaids_dir, "Annotations_RGB/"),
                    os.path.join(mobilityaids_dir, "Annotations_RGB_TestSet2/"),
                    os.path.join(mobilityaids_dir, "Annotations_RGB_TestSet2/"),
                    os.path.join(mobilityaids_dir, "Annotations_RGB_TestSet2/"),
                    os.path.join(mobilityaids_dir, "Annotations_RGB_TestSet2/")]

    image_dirs = [os.path.join(mobilityaids_dir, "Images_DepthJet/"),
                os.path.join(mobilityaids_dir, "Images_RGB/")]


    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples variable

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == "__main__":
    tf.app.run()
