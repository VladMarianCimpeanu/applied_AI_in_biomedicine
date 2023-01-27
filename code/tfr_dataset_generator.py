import os
import PIL
import sys
import io

from math import ceil
from PIL import Image



class ProgressBar:

    def __init__(self, end, width=15, step_size=1) -> None:
        """
        This class implements a dynamic progress bar.
        :param end: number of iteration of the process to represent.
        :param width: width of the diplayed progress bar. Default is 15.
        :param step_size: size of the steps at each iteration. By default it is set to 1.
        """
        self.step = 0
        self.end = end
        self.width = width
        self.step_size = step_size

    def reset(self):
        """
        reset the progress bar to the initial state.
        :return: None
        """
        self.__init__(self.end, self.width, self.step_size)

    def next(self):
        """
        print updated progress bar.
        :return: None
        """
        self.step += self.step_size
        percentage = self.step / self.end * 100
        n_completed = ceil(percentage / 100 * self.width)
        completed = "=" * n_completed
        to_complete = " " * (self.width - n_completed)
        sys.stdout.write("\rloading: [{}{}] {:0.1f}%".format(
            completed, to_complete, percentage))
        if self.step == self.end:
            print()


def build_TFRecord(labels, paths, name, folder_path='TFRdataset', size=256):
    """
    Given am image dataset, this function create a new folder containg the new datataset
    with resized images and TFRrecord format. 
    :param labels: array of sparse labels
    :param paths: array whose i-th element is the path of the i-th image with i-th label of the labels array.
    :param folder_path: root name folder of the new dataset.
    :param name: name of the dataset. subdirectory name (used to divide the dataset into train, validation and test).
    :param size: new size of the images.
    :return: None.
    """
    if not os.path.exists(f'./{folder_path}/'):
        os.mkdir(f'./{folder_path}/')
    
    # create new tfrecord file 
    with tf.io.TFRecordWriter(f"./{folder_path}/{name}.tfrecords") as writer:
        progressbar = ProgressBar(len(labels))
        for path, label in zip(paths, labels):
            progressbar.next()
            image = Image.open(path)
            # resize image
            image = image.resize((size, size))
            bytes_buffer = io.BytesIO()
            # compress image using jpeg format.
            image.convert("L").save(bytes_buffer, "JPEG")
            image_bytes = bytes_buffer.getvalue()

            bytes_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
            class_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            
            # create tfrecord for the current image
            example = tf.train.Example(
              features=tf.train.Features(feature={
                  "image": bytes_feature,
                  "class": class_feature
              })
            )

            writer.write(example.SerializeToString())

            image.close()
        progressbar.reset()


if __name__ == "__main__":
    images_paths = [[], [], []]
    labels = [[], [], []]
    sparse_labels = [[], [], []]

    map_set_to_indx = {
        'training': 0,
        'validation': 1,
        'test': 2
    }

    map_label_to_int = {
        'n': 0,
        'p': 1,
        't': 2
    }

    for root_dir, cur_dir, files in os.walk('./dataset'): # <-------- change this line for the original dataset directory.
        # if files is empty pass, there is nothing to do.
        if files:
            splitted_root_dir = root_dir.split('\\') 
            set_destination = splitted_root_dir[1]
            label = splitted_root_dir[2]
            for item in files:
                labels[map_set_to_indx[set_destination]].append(label)
                images_paths[map_set_to_indx[set_destination]].append(f'{root_dir}\\{item}')
                sparse_labels[map_set_to_indx[set_destination]].append(map_label_to_int[label])

    build_TFRecord(sparse_labels[map_set_to_indx['training']],
               images_paths[map_set_to_indx['training']],
               'training')

    build_TFRecord(sparse_labels[map_set_to_indx['validation']],
               images_paths[map_set_to_indx['validation']],
               'validation')

    build_TFRecord(sparse_labels[map_set_to_indx['test']],
               images_paths[map_set_to_indx['test']],
               'test')