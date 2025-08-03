import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Upsampling.model import Model
from Upsampling.configs import FLAGS
from datetime import datetime
import os
from extract_powerline.xyz2h5 import make_clean_folder
from extract_powerline.copy_xyz import copy_xyz_files
import logging
import pprint
import shutil
from extract_powerline.merge_xyz import merge_xyz_files
pp = pprint.PrettyPrinter()

def run():
    if FLAGS.phase=='train':
        FLAGS.train_file = os.path.join(FLAGS.data_dir, 'train/train_512_2048.h5')
        print('train_file:',FLAGS.train_file)
        if not FLAGS.restore:
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            FLAGS.log_dir = os.path.join(FLAGS.log_dir,current_time)
            try:
                os.makedirs(FLAGS.log_dir)
            except os.error:
                pass
    else:
        FLAGS.log_dir = os.path.join(os.getcwd(),'model')
        FLAGS.test_data = os.path.join(FLAGS.data_dir, 'test/*.xyz')
        FLAGS.out_folder = os.path.join(FLAGS.data_dir,'test/output')
        if not os.path.exists(FLAGS.out_folder):
            os.makedirs(FLAGS.out_folder)
        print('test_data:',FLAGS.test_data)

    print('checkpoints:',FLAGS.log_dir)
    pp.pprint(FLAGS)
    # open session
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        model = Model(FLAGS,sess)
        if FLAGS.phase == 'train':
            model.train()
        else:
            model.test()
            # merge results
            input_folder = "data/test/output"
            output_folder = "data/test/temp"
            merge_xyz_files(input_folder, os.path.join(output_folder, "result.xyz"))
            make_clean_folder(input_folder)
            copy_xyz_files(output_folder, input_folder)
            


def main(unused_argv):
  run()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
