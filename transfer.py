import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from additional import load_and_process_img
from additional import get_style_loss
from additional import compute_loss
from additional import get_model
from additional import get_feature_representations
from additional import compute_grads
from additional import deprocess_img
from additional import get_content_loss
from additional import gram_matrix

content_path = 'resource\YellowLabradorLooking_new.jpg'
style_path = 'resource\The_Great_Wave_off_Kanagawa.jpg'

def run_style_transfer(content_path, style_path, num_iterations=1000, content_weight=1e3, style_weight=1e-2):
  model = get_model()

  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

  init_image = load_and_process_img(content_path)
  init_image = tf.Variable(init_image, dtype=tf.float32)

  opt = tf.compat.v1.train.AdamOptimizer(learning_rate=5, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')

  best_loss, best_img = float('inf'), None


  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }

  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means

  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)

    if loss < best_loss:
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())

  return best_img, best_loss

best, best_loss = run_style_transfer(content_path, style_path, num_iterations=1000)

from PIL import Image
Image.fromarray(best).save('resource\output.jpg')
