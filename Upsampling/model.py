# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:04 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Upsampling.generator import Generator
from Upsampling.discriminator import Discriminator
from Common.visu_utils import plot_pcd_three_views,point_cloud_three_views
from Common.ops import add_scalar_summary,add_hist_summary
from Upsampling.data_loader import Fetcher
from Common import model_utils
from Common import pc_util
from Common.loss_utils import pc_distance,get_uniform_loss,get_repulsion_loss,discriminator_loss,generator_loss, line_fit_loss
from tf_ops.sampling.tf_sampling import farthest_point_sample
import logging
import os
from tqdm import tqdm
from glob import glob
import math
from time import time
from termcolor import colored
import numpy as np
from Common.pc_util import group_wires_general, extract_segments_per_wire_general
from sklearn.decomposition import PCA
import csv

class Model(object):
  def __init__(self,opts,sess):
      self.sess = sess
      self.opts = opts

  def allocate_placeholders(self):
      self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
      self.input_x = tf.placeholder(tf.float32, shape=[self.opts.batch_size,self.opts.num_point,3])
      self.input_y = tf.placeholder(tf.float32, shape=[self.opts.batch_size, int(self.opts.up_ratio*self.opts.num_point),3])
      self.pc_radius = tf.placeholder(tf.float32, shape=[self.opts.batch_size])

  def build_model(self):
      self.G = Generator(self.opts,self.is_training,name='generator')
      self.D = Discriminator(self.opts, self.is_training, name='discriminator')

      # X -> Y
      self.G_y = self.G(self.input_x)
      self.dis_loss = self.opts.fidelity_w * pc_distance(self.G_y, self.input_y, radius=self.pc_radius)

      if self.opts.use_repulse:
          self.repulsion_loss = self.opts.repulsion_w*get_repulsion_loss(self.G_y)
      else:
          self.repulsion_loss = 0
      self.uniform_loss = self.opts.uniform_w * get_uniform_loss(self.G_y)
    ### thêm line fit loss ############################
    #   self.line_fit_loss = self.opts.line_fit_w * line_fit_loss(self.G_y)
    #   self.pu_loss = (self.dis_loss + self.uniform_loss + self.repulsion_loss +
    #             self.line_fit_loss + tf.losses.get_regularization_loss())

      self.pu_loss = (self.dis_loss + self.uniform_loss + self.repulsion_loss + tf.losses.get_regularization_loss())
    #################################
      self.G_gan_loss = self.opts.gan_w*generator_loss(self.D,self.G_y)
      self.total_gen_loss = self.G_gan_loss + self.pu_loss

      self.D_loss = discriminator_loss(self.D,self.input_y,self.G_y)

      self.setup_optimizer()
      self.summary_all()

      self.visualize_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
      self.visualize_titles = ['input_x', 'fake_y', 'real_y']

  def summary_all(self):

      # summary
      add_scalar_summary('loss/dis_loss', self.dis_loss, collection='gen')
      add_scalar_summary('loss/repulsion_loss', self.repulsion_loss,collection='gen')
      add_scalar_summary('loss/uniform_loss', self.uniform_loss,collection='gen')
      add_scalar_summary('loss/G_loss', self.G_gan_loss,collection='gen')
      add_scalar_summary('loss/total_gen_loss', self.total_gen_loss, collection='gen')

      add_hist_summary('D/true',self.D(self.input_y),collection='dis')
      add_hist_summary('D/fake',self.D(self.G_y),collection='dis')
      add_scalar_summary('loss/D_Y', self.D_loss,collection='dis')

      self.g_summary_op = tf.summary.merge_all('gen')
      self.d_summary_op = tf.summary.merge_all('dis')

      self.visualize_x_titles = ['input_x', 'fake_y', 'real_y']
      self.visualize_x_ops = [self.input_x[0], self.G_y[0], self.input_y[0]]
      self.image_x_merged = tf.placeholder(tf.float32, shape=[None, 1500, 1500, 1])
      self.image_x_summary = tf.summary.image('Upsampling', self.image_x_merged, max_outputs=1)

  def setup_optimizer(self):

      learning_rate_d = tf.where(
          tf.greater_equal(self.global_step, self.opts.start_decay_step),
          tf.train.exponential_decay(self.opts.base_lr_d, self.global_step - self.opts.start_decay_step,
                                    self.opts.lr_decay_steps, self.opts.lr_decay_rate,staircase=True),
          self.opts.base_lr_d
      )
      learning_rate_d = tf.maximum(learning_rate_d, self.opts.lr_clip)
      add_scalar_summary('learning_rate/learning_rate_d', learning_rate_d,collection='dis')


      learning_rate_g = tf.where(
          tf.greater_equal(self.global_step, self.opts.start_decay_step),
          tf.train.exponential_decay(self.opts.base_lr_g, self.global_step - self.opts.start_decay_step,
                                     self.opts.lr_decay_steps, self.opts.lr_decay_rate, staircase=True),
          self.opts.base_lr_g
      )
      learning_rate_g = tf.maximum(learning_rate_g, self.opts.lr_clip)
      add_scalar_summary('learning_rate/learning_rate_g', learning_rate_g, collection='gen')

      # create pre-generator ops
      gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
      gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

      with tf.control_dependencies(gen_update_ops):
          self.G_optimizers = tf.train.AdamOptimizer(learning_rate_g, beta1=self.opts.beta).minimize(self.total_gen_loss, var_list=gen_tvars,
                                                                                              colocate_gradients_with_ops=True,
                                                                                              global_step=self.global_step)

      self.D_optimizers = tf.train.AdamOptimizer(learning_rate_d, beta1=self.opts.beta).minimize(self.D_loss,
                                                                                                 self.global_step,
                                                                                                 var_list=self.D.variables,
                                                                                                 name='Adam_D_X')

  def train(self):

      self.allocate_placeholders()
      self.build_model()

      self.sess.run(tf.global_variables_initializer())

      fetchworker = Fetcher(self.opts)
      fetchworker.start()

      self.saver = tf.train.Saver(max_to_keep=None)
      self.writer = tf.summary.FileWriter(self.opts.log_dir, self.sess.graph)

      restore_epoch = 0
      if self.opts.restore:
          restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
          self.saver.restore(self.sess, checkpoint_path)
          #self.saver.restore(self.sess, tf.train.latest_checkpoint(self.opts.log_dir))
          self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'a')
          tf.assign(self.global_step, restore_epoch * fetchworker.num_batches).eval()
          restore_epoch += 1

      else:
          os.makedirs(os.path.join(self.opts.log_dir, 'plots'))
          self.LOG_FOUT = open(os.path.join(self.opts.log_dir, 'log_train.txt'), 'w')

      with open(os.path.join(self.opts.log_dir, 'args.txt'), 'w') as log:
          for arg in sorted(vars(self.opts)):
              log.write(arg + ': ' + str(getattr(self.opts, arg)) + '\n')  # log of arguments

      step = self.sess.run(self.global_step)
      start = time()
      for epoch in range(restore_epoch, self.opts.training_epoch):
          logging.info('**** EPOCH %03d ****\t' % (epoch))
          for batch_idx in range(fetchworker.num_batches):

              batch_input_x, batch_input_y,batch_radius = fetchworker.fetch()

              feed_dict = {self.input_x: batch_input_x,
                           self.input_y: batch_input_y,
                           self.pc_radius: batch_radius,
                           self.is_training: True}


              # Update D network
              _,d_loss,d_summary = self.sess.run([self.D_optimizers,self.D_loss,self.d_summary_op],feed_dict=feed_dict)
              self.writer.add_summary(d_summary, step)

              # Update G network
              for i in range(self.opts.gen_update):
                  # get previously generated images
                  _, g_total_loss, summary = self.sess.run(
                      [self.G_optimizers, self.total_gen_loss, self.g_summary_op], feed_dict=feed_dict)
                  self.writer.add_summary(summary, step)

              if step % self.opts.steps_per_print == 0:
                  self.log_string('-----------EPOCH %d Step %d:-------------' % (epoch,step))
                  self.log_string('  G_loss   : {}'.format(g_total_loss))
                  self.log_string('  D_loss   : {}'.format(d_loss))
                  self.log_string(' Time Cost : {}'.format(time() - start))
                  start = time()
                  feed_dict = {self.input_x: batch_input_x,
                               self.is_training: False}

                  fake_y_val = self.sess.run([self.G_y], feed_dict=feed_dict)


                  fake_y_val = np.squeeze(fake_y_val)
                  image_input_x = point_cloud_three_views(batch_input_x[0])
                  image_fake_y = point_cloud_three_views(fake_y_val[0])
                  image_input_y = point_cloud_three_views(batch_input_y[0, :, 0:3])
                  image_x_merged = np.concatenate([image_input_x, image_fake_y, image_input_y], axis=1)
                  image_x_merged = np.expand_dims(image_x_merged, axis=0)
                  image_x_merged = np.expand_dims(image_x_merged, axis=-1)
                  image_x_summary = self.sess.run(self.image_x_summary, feed_dict={self.image_x_merged: image_x_merged})
                  self.writer.add_summary(image_x_summary, step)

              if self.opts.visulize and (step % self.opts.steps_per_visu == 0):
                  feed_dict = {self.input_x: batch_input_x,
                               self.input_y: batch_input_y,
                               self.pc_radius: batch_radius,
                               self.is_training: False}
                  pcds = self.sess.run([self.visualize_ops], feed_dict=feed_dict)
                  pcds = np.squeeze(pcds)  # np.asarray(pcds).reshape([3,self.opts.num_point,3])
                  plot_path = os.path.join(self.opts.log_dir, 'plots',
                                           'epoch_%d_step_%d.png' % (epoch, step))
                  plot_pcd_three_views(plot_path, pcds, self.visualize_titles)

              step += 1
          if (epoch % self.opts.epoch_per_save) == 0:
              self.saver.save(self.sess, os.path.join(self.opts.log_dir, 'model'), epoch)
              print(colored('Model saved at %s' % self.opts.log_dir, 'white', 'on_blue'))

      fetchworker.shutdown()

  def patch_prediction(self, patch_point):
      # normalize the point clouds
      patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
      patch_point = np.expand_dims(patch_point, axis=0)
      pred = self.sess.run([self.pred_pc], feed_dict={self.inputs: patch_point})
      pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
      return pred
  

#   def pc_prediction(self, pc):
#     """
#     Chia patch liên tiếp dọc theo trục chính của dây (chỉ 1 dây).
#     - Xác định trục chính bằng PCA.
#     - Chia patch liên tiếp dọc trục chính.
#     """
#     # 1. PCA để xác định trục chính
#     pca = PCA(n_components=1)
#     proj_coords = pca.fit_transform(pc).flatten()
#     min_proj, max_proj = proj_coords.min(), proj_coords.max()

#     patch_num_point = self.opts.patch_num_point
#     patch_length = patch_num_point  # số điểm mỗi patch (có thể tùy chỉnh)
#     stride = patch_num_point  # nếu muốn overlap, giảm stride < patch_num_point

#     # 2. Sắp xếp theo giá trị trên trục chính
#     sort_idx = np.argsort(proj_coords)
#     pc_sorted = pc[sort_idx]
#     num_points = pc_sorted.shape[0]

#     # 3. Chia patch liên tiếp (không overlap)
#     patches = []
#     for start in range(0, num_points, stride):
#         end = start + patch_length
#         if end > num_points:
#             # Nếu patch cuối ít điểm hơn, bỏ qua hoặc có thể lấy toàn bộ phần còn lại
#             break
#         patch = pc_sorted[start:end]
#         patches.append(patch)

#     input_list = []
#     up_point_list = []
#     for point in tqdm(patches, total=len(patches)):
#         up_point = self.patch_prediction(point)
#         up_point = np.squeeze(up_point, axis=0)
#         input_list.append(point)
#         up_point_list.append(up_point)

#     return input_list, up_point_list

  def pc_prediction(self, segment1, segment2):
    """
    Chỉ lấy patch gồm opts.patch_num_point điểm:
    - opts.patch_num_point//2 điểm cuối của segment1
    - opts.patch_num_point//2 điểm đầu của segment2
    - Nếu tổng số điểm < opts.patch_num_point: giữ nguyên, không upsample, đưa vào output
    - Nếu một bên không đủ, lấy bù bên còn lại cho đủ patch_num_point
    Trả về input_list, up_point_list (mỗi list có 1 phần tử)
    """
    patch_num_point = self.opts.patch_num_point
    half = patch_num_point // 2

    len1 = segment1.shape[0]
    len2 = segment2.shape[0]

    input_list = []
    up_point_list = []

    # Trường hợp tổng số điểm < patch_num_point: giữ nguyên, không upsample
    if len1 + len2 < patch_num_point:
        patch_orig = np.concatenate([segment1, segment2], axis=0)
        input_list.append(patch_orig)
        up_point_list.append(patch_orig)
        return input_list, up_point_list

    # Trường hợp một bên không đủ half, lấy bù bên còn lại
    if len1 >= half and len2 >= half:
        pts1 = segment1[-half:]
        pts2 = segment2[:half]
    elif len1 < half:
        pts1 = segment1
        pts2 = segment2[:(patch_num_point - len1)]
    elif len2 < half:
        pts1 = segment1[-(patch_num_point - len2):]
        pts2 = segment2
    patch = np.concatenate([pts1, pts2], axis=0)

    # Kiểm tra lại lần nữa (bất khả kháng)
    if patch.shape[0] != patch_num_point:
        input_list.append(np.concatenate([segment1, segment2], axis=0))
        up_point_list.append(np.concatenate([segment1, segment2], axis=0))
        return input_list, up_point_list

    # Gọi upsample
    up_point = self.patch_prediction(patch)
    up_point = np.squeeze(up_point, axis=0)
    input_list.append(patch)
    up_point_list.append(up_point)
    return input_list, up_point_list
  

#   def test(self):
#     self.inputs = tf.placeholder(tf.float32, shape=[1, self.opts.patch_num_point, 3])
#     is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
#     Gen = Generator(self.opts, is_training, name='generator')
#     #kiểm tra xem có cần upsample không (nếu mật độ điểm đầu vào đã đủ lớn thì không cần upsample)
    
#     self.pred_pc = Gen(self.inputs)
#     for i in range(round(math.pow(self.opts.up_ratio, 1 / 4)) - 1):
#         self.pred_pc = Gen(self.pred_pc)

#     saver = tf.train.Saver()
#     restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
#     print(checkpoint_path)
#     saver.restore(self.sess, checkpoint_path)

#     samples = glob(self.opts.test_data)

#     for point_path in samples:
#         logging.info(point_path)
#         start = time()
#         pc = pc_util.load(point_path)[:, :3]
#         pc, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)

#         if self.opts.jitter:
#             pc = pc_util.jitter_perturbation_point_cloud(
#                 pc[np.newaxis, ...], 
#                 sigma=self.opts.jitter_sigma,
#                 clip=self.opts.jitter_max
#             )
#             pc = pc[0, ...]

#         input_list, pred_list = self.pc_prediction(pc)

#         end = time()
#         print("total time: ", end - start)

#         if len(pred_list) == 0:
#             logging.warning(f"No valid patch for {point_path}, pred_list is empty! Skipping.")
#             continue

#         pred_pc = np.concatenate(pred_list, axis=0)
#         pred_pc = (pred_pc * furthest_distance) + centroid

#         pred_pc = np.reshape(pred_pc, [-1, 3])
#         # --- Sửa tại đây: lấy số điểm input cho từng file ---
#         out_point_num = int(pc.shape[0] * self.opts.up_ratio)
#         path = os.path.join(self.opts.out_folder, point_path.split('/')[-1][:-4] + '.ply')
#         idx = farthest_point_sample(out_point_num, pred_pc[np.newaxis, ...]).eval()[0]
#         pred_pc = pred_pc[idx, 0:3]
#         np.savetxt(path[:-4] + '.xyz', pred_pc, fmt='%.6f')

  def test(self):
    self.inputs = tf.placeholder(tf.float32, shape=[1, self.opts.patch_num_point, 3])
    is_training = tf.placeholder_with_default(False, shape=[], name='is_training')
    Gen = Generator(self.opts, is_training, name='generator')
    self.pred_pc = Gen(self.inputs)
    for i in range(round(math.pow(self.opts.up_ratio, 1 / 4)) - 1):
        self.pred_pc = Gen(self.pred_pc)

    saver = tf.train.Saver()
    restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(self.opts.log_dir)
    print(checkpoint_path)
    saver.restore(self.sess, checkpoint_path)

    # ---- Đọc match_segment.csv bằng csv.reader ----
    match_csv = self.opts.match_segment_csv
    bridge_pairs = []
    with open(match_csv, "r", newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Bỏ qua header
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 7:
                continue
            seg1, len1, seg2, len2 = row[0], float(row[1]), row[2], float(row[3])
            bridge_pairs.append((seg1, seg2))
    print(f"Found {len(bridge_pairs)} bridge pairs from {match_csv}.")

    # ---- Nạp tất cả segment vào dict để tra nhanh ----
    segment_folder = self.opts.segment_folder
    segment_files = {os.path.basename(f): f for f in glob(os.path.join(segment_folder, "*.xyz"))}
    point_dict = {name: np.loadtxt(f) for name, f in segment_files.items()}
    print(f"Loaded {len(point_dict)} segments from {segment_folder}")

    # ---- Xử lý từng cặp cần upsample bridge ----
    for seg1_name, seg2_name in bridge_pairs:
        print(f"Processing pair: {seg1_name} -> {seg2_name}")
        if seg1_name not in point_dict or seg2_name not in point_dict:
            logging.warning(f"Missing segment {seg1_name} or {seg2_name}, skip.")
            continue
        seg1 = point_dict[seg1_name]
        seg2 = point_dict[seg2_name]
        input_list, pred_list = self.pc_prediction(seg1, seg2)  # pc_prediction nhận 2 segment (đã chỉnh sửa!)

        if len(pred_list) == 0:
            logging.warning(f"No valid patch for {seg1_name}, {seg2_name}!")
            continue

        pred_pc = np.concatenate(pred_list, axis=0)
        # Lưu file kết quả
        out_xyz = os.path.join(self.opts.out_folder, f"upsampled_{seg1_name}_to_{seg2_name}.xyz")
        np.savetxt(out_xyz, pred_pc, fmt='%.6f')
        print(f"Saved upsampled bridge: {out_xyz}")


  def log_string(self,msg):
      #global LOG_FOUT
      logging.info(msg)
      self.LOG_FOUT.write(msg + "\n")
      self.LOG_FOUT.flush()











