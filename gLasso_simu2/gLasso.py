from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv
import simu_data
import numbers
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
import tensorlayer as tl


def group_regularizer(scale, scope=None):

    if isinstance(scale, numbers.Integral):
        raise ValueError('scale cannot be an integer: %s' % scale)
    if isinstance(scale, numbers.Real):
        if scale < 0.:
            raise ValueError('Setting a scale less than 0 on a regularizer: %g' % scale)
        if scale >= 1.:
            raise ValueError('Setting a scale greater than 1 on a regularizer: %g' % scale)
        if scale == 0.:
            tl.logging.info('Scale of 0 disables regularizer.')
            return lambda _, name=None: None

    def group(weights):
        """Applies group regularization to weights."""
        with tf.name_scope('group_regularizer') as scope:
            my_scale = ops.convert_to_tensor(scale, dtype=weights.dtype.base_dtype, name='scale')
            standard_ops_fn = standard_ops.multiply
            return standard_ops_fn(
                my_scale,
                standard_ops.reduce_sum(standard_ops.sqrt(standard_ops.reduce_sum(tf.square(weights), 1))),
                name=scope
            )

    return group

def cosine_distance(v1, v2):
    return tf.reduce_sum(tf.multiply(v1, v2), 1) / \
        (tf.sqrt(tf.reduce_sum(tf.multiply(v1, v1), 1)) *
         tf.sqrt(tf.reduce_sum(tf.multiply(v2, v2), 1)))

def schatten2_norm(x):
    return np.sqrt(np.sum(x**2))

def gLasso_model(features, labels, mode, params):

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    if params['hidden_units'][0] == 0:
        regularizer = tf.contrib.layers.l1_regularizer(scale=params['reg'])
        response = tf.layers.dense(net, params['n_response'],
                                   activation=None, kernel_regularizer=regularizer)
    else:
        regularizer = group_regularizer(scale=params['reg'])
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

# def main(argv):
#
#     # Fetch the data
#     (train_x, train_y), (test_x, test_y) = simu_data.load_data()
#
#     # Feature columns describe how to use the input.
#     my_feature_columns = []
#     for key in train_x.keys():
#         my_feature_columns.append(
#             tf.feature_column.numeric_column(key=key))
#
#     # Build 2 hidden layer DNN with 20, 20 units respectively.
#     classifier = tf.estimator.Estimator(
#         model_fn=gLasso_model,
#         params={
#             'feature_columns': my_feature_columns,
#             # Two hidden layers of 20 nodes each.
#             'hidden_units': [3, 2],
#             # The model output.
#             'n_response': 1,
#             # lambda.
#             'reg': reg,
#         })
#
#     # Train the Model.
#     classifier.train(
#         input_fn=lambda: simu_data.train_input_fn(
#             train_x, train_y, 100),
#         steps=1500)
#
#     # Evaluate the model.
#     eval_result = classifier.evaluate(
#         input_fn=lambda: simu_data.eval_input_fn(test_x, test_y, 100))
#
#     # extract variables from model.
#     var_dict = dict()
#     for var_name in classifier.get_variable_names():
#         var_dict[var_name] = classifier.get_variable_value(var_name)
#
#     spec_norm = np.linalg.norm(var_dict["dense/kernel"], 2) + \
#                np.linalg.norm(var_dict["dense_1/kernel"], 2) + \
#                np.linalg.norm(var_dict["dense_2/kernel"], 2)
#
#     u0, s0, vh0 = np.linalg.svd(var_dict["dense/kernel"])
#     u1, s1, vh1 = np.linalg.svd(var_dict["dense_1/kernel"])
#     u2, s2, vh2 = np.linalg.svd(var_dict["dense_2/kernel"])
#     hs_norm = schatten2_norm(s0) + schatten2_norm(s1) + schatten2_norm(s2)
#
#     # feature_weight = classifier.get_variable_value('Variable')
#     # layer1_weight = classifier.get_variable_value('dense/kernel')
#
#     print('\nTest set MSE: {MSE:0.3f}\n'.format(**eval_result))
#     print('\nSpectral Norm: \n', spec_norm)
#     print('\nHilbert–Schmidt Norm: \n', hs_norm)
#     # print('\nWeights of first layer: \n', feature_weight)
#     # print('\nWeights of output layer: \n', layer1_weight)
#     print('\nWeights of output layer: \n', var_dict)


# if __name__ == '__main__':
#     tf.logging.set_verbosity(tf.logging.INFO)
#     tf.app.run(main)

N_runs = 20
reg_factors = np.linspace(0, 1, 10)
n_list = np.array([100, 200, 500, 1000])
method_list = ["linear", "polynomial", "rbf"]
snr_list = np.array([2, 1])
rho_list = np.array([0, .2])
p_list = np.array([20, 50, 100])

## layers = 2
csvFile2 = open("result2.csv", "a")
fileHeader = ["Nk", "n", "method", "snr", "rho", "p", "mse", "spec_norm", "hs_norm"]
writer = csv.writer(csvFile2)
writer.writerow(fileHeader)
for k in np.arange(0, N_runs):
    print("Run ", k + 1, " of ", N_runs, "...\n", end="")
    for n in n_list:
        for method in method_list:
            for snr in snr_list:
                for rho in rho_list:
                    for p in p_list:
                        for reg in reg_factors:
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
                                    'hidden_units': [20, 20],
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

                            spec_norm = np.linalg.norm(var_dict["dense/kernel"], 2) + \
                                        np.linalg.norm(var_dict["dense_1/kernel"], 2) + \
                                        np.linalg.norm(var_dict["dense_2/kernel"], 2)

                            u0, s0, vh0 = np.linalg.svd(var_dict["dense/kernel"])
                            u1, s1, vh1 = np.linalg.svd(var_dict["dense_1/kernel"])
                            u2, s2, vh2 = np.linalg.svd(var_dict["dense_2/kernel"])
                            hs_norm = schatten2_norm(s0) + schatten2_norm(s1) + schatten2_norm(s2)

                            # feature_weight = classifier.get_variable_value('Variable')
                            # layer1_weight = classifier.get_variable_value('dense/kernel')

                            print('\nTest set MSE: {MSE:0.3f}\n'.format(**eval_result))
                            print('\nSpectral Norm: \n', spec_norm)
                            print('\nHilbert–Schmidt Norm: \n', hs_norm)
                            # print('\nWeights of first layer: \n', feature_weight)
                            # print('\nWeights of output layer: \n', layer1_weight)
                            print('\nWeights of output layer: \n', var_dict)

                            add_info = [k, n, method, snr, rho, p, eval_result['MSE'], spec_norm, hs_norm]
                            writer.writerow(add_info)
csvFile2.close()



## layers = 1
csvFile1 = open("result1.csv", "a")
fileHeader = ["Nk", "n", "method", "snr", "rho", "p", "mse", "spec_norm", "hs_norm"]
writer = csv.writer(csvFile1)
writer.writerow(fileHeader)
for k in np.arange(0, N_runs):
    print("Run ", k + 1, " of ", N_runs, "...\n", end="")
    for n in n_list:
        for method in method_list:
            for snr in snr_list:
                for rho in rho_list:
                    for p in p_list:
                        for reg in reg_factors:
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

                            spec_norm = np.linalg.norm(var_dict["dense/kernel"], 2) + \
                                        np.linalg.norm(var_dict["dense_1/kernel"], 2)

                            u0, s0, vh0 = np.linalg.svd(var_dict["dense/kernel"])
                            u1, s1, vh1 = np.linalg.svd(var_dict["dense_1/kernel"])
                            hs_norm = schatten2_norm(s0) + schatten2_norm(s1)

                            # feature_weight = classifier.get_variable_value('Variable')
                            # layer1_weight = classifier.get_variable_value('dense/kernel')

                            print('\nTest set MSE: {MSE:0.3f}\n'.format(**eval_result))
                            print('\nSpectral Norm: \n', spec_norm)
                            print('\nHilbert–Schmidt Norm: \n', hs_norm)
                            # print('\nWeights of first layer: \n', feature_weight)
                            # print('\nWeights of output layer: \n', layer1_weight)
                            print('\nWeights of output layer: \n', var_dict)

                            add_info = [k, n, method, snr, rho, p, eval_result['MSE'], spec_norm, hs_norm]
                            writer.writerow(add_info)
csvFile1.close()


## layers = 0
csvFile0 = open("result0.csv", "a")
fileHeader = ["Nk", "n", "method", "snr", "rho", "p", "mse", "spec_norm", "hs_norm"]
writer = csv.writer(csvFile0)
writer.writerow(fileHeader)
for k in np.arange(0, N_runs):
    print("Run ", k + 1, " of ", N_runs, "...\n", end="")
    for n in n_list:
        for method in method_list:
            for snr in snr_list:
                for rho in rho_list:
                    for p in p_list:
                        for reg in reg_factors:
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
                                    'hidden_units': [0],
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

                            spec_norm = np.linalg.norm(var_dict["dense/kernel"], 2)

                            u0, s0, vh0 = np.linalg.svd(var_dict["dense/kernel"])
                            hs_norm = schatten2_norm(s0)

                            # feature_weight = classifier.get_variable_value('Variable')
                            # layer1_weight = classifier.get_variable_value('dense/kernel')

                            print('\nTest set MSE: {MSE:0.3f}\n'.format(**eval_result))
                            print('\nSpectral Norm: \n', spec_norm)
                            print('\nHilbert–Schmidt Norm: \n', hs_norm)
                            # print('\nWeights of first layer: \n', feature_weight)
                            # print('\nWeights of output layer: \n', layer1_weight)
                            print('\nWeights of output layer: \n', var_dict)

                            add_info = [k, n, method, snr, rho, p, eval_result['MSE'], spec_norm, hs_norm]
                            writer.writerow(add_info)
csvFile0.close()
