from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv
import simu_data
import numbers
import argparse
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
import tensorlayer as tl
from tensorflow.python.platform import tf_logging as logging
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument("n", type=int, help="sample size")
parser.add_argument("rho", type=float, help="correlation coefficient")
parser.add_argument("p", type=int, help="dimension of features")
args = parser.parse_args()
n = args.n
rho = args.rho
p = args.p

def group2_regularizer(scale, scope=None):

  if isinstance(scale, numbers.Integral):
    raise ValueError('scale cannot be an integer: %s' % scale)
  if isinstance(scale, numbers.Real):
    if scale < 0.:
      raise ValueError('Setting a scale less than 0 on a regularizer: %g' %
                       scale)
    if scale == 0.:
      logging.info('Scale of 0 disables regularizer.')
      return lambda _: None

  def group2(weights, name=None):
    """Applies group regularization to weights."""
    with ops.name_scope(scope, 'group2_regularizer', [weights]) as name:
      my_scale = ops.convert_to_tensor(scale,
                                       dtype=weights.dtype.base_dtype,
                                       name='scale')
      return standard_ops.multiply(
          my_scale,
          standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 1))),
          name=name)

  return group2


def gLasso_model(features, labels, mode, params):

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    if params['hidden_units'][0] == 0:
        regularizer = tf.contrib.layers.l1_regularizer(scale=params['reg'])
        response = tf.layers.dense(net, params['n_response'],
                                   activation=None, kernel_regularizer=regularizer)
    else:
        regularizer = group2_regularizer(scale=params['reg'])
        net = tf.layers.dense(net, units=params['hidden_units'][0],
                              activation=tf.nn.relu, kernel_regularizer=regularizer)
        if len(params['hidden_units']) >= 2:
            for units in params['hidden_units'][1:]:
                net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        response = tf.layers.dense(net, params['n_response'], activation=None)

    response = tf.squeeze(response)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "response": response,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=response)

    # Compute evaluation metrics.
    mse = tf.metrics.mean_squared_error(labels=labels,
                                        predictions=response)
    metrics = {'MSE': mse}
    tf.summary.scalar("MSE", mse[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


N_runs = 20
reg_factors = np.e**np.linspace(-10, 7, 10)
# n_list = np.array([100, 200, 500, 1000])
method_list = ["linear", "polynomial", "rbf"]
# snr_list = np.array([2, 1.5])
# rho_list = np.array([0, .2])
# p_list = np.array([20, 50, 100])
snr = 2

## layers = 1
csvFile1 = open("result1.csv", "a")
fileHeader = ["Nk", "n", "method", "snr", "rho", "p", "reg", "cos_dis", "mse", "spec_norm", "hs_norm"]
writer = csv.writer(csvFile1)
writer.writerow(fileHeader)

for method in method_list:
    for reg in reg_factors:
        for k in np.arange(0, N_runs):
            print("Run ", k + 1, " of ", N_runs, "...\n", end="")
            (train_x, train_y), (test_x, test_y) = \
                simu_data.load_data(y_name="y", n=n,
                                    method=method, rho=rho, snr=snr, p=p)
            my_feature_columns = []
            for key in train_x.keys():
                my_feature_columns.append(
                    tf.feature_column.numeric_column(key=key))

            classifier = tf.estimator.Estimator(
                model_fn=gLasso_model,
                params={
                    'feature_columns': my_feature_columns,
                    'hidden_units': [20],
                    'n_response': 1,
                    'reg': reg,
                })

            classifier.train(
                input_fn=lambda: simu_data.train_input_fn(
                    train_x, train_y, 100),
                steps=1500)

            eval_result = classifier.evaluate(
                input_fn=lambda: simu_data.eval_input_fn(test_x, test_y, 100))

            var_dict = dict()
            for var_name in classifier.get_variable_names():
                var_dict[var_name] = classifier.get_variable_value(var_name)

            v1 = np.zeros(p)
            v1[0] = v1[3] = v1[6] = 1
            v1 = v1.reshape((1, -1))
            v2 = np.linalg.norm(var_dict["dense/kernel"], axis=1).reshape((1, -1))
            cos_dis = cosine_similarity(v1, v2)
            cos_dis = cos_dis[0][0]

            spec_norm = np.linalg.norm(var_dict["dense/kernel"], 2) + \
                        np.linalg.norm(var_dict["dense_1/kernel"], 2)

            u0, s0, vh0 = np.linalg.svd(var_dict["dense/kernel"])
            u1, s1, vh1 = np.linalg.svd(var_dict["dense_1/kernel"])
            hs_norm = np.linalg.norm(s0) + np.linalg.norm(s1)

            add_info = [k, n, method, snr, rho, p, reg, cos_dis, eval_result['MSE'], spec_norm, hs_norm]
            writer.writerow(add_info)
            print(add_info)
csvFile1.close()

