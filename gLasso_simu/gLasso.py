from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import numpy as np
import csv
import simu_data

# parser = argparse.ArgumentParser()
# parser.add_argument('--batch_size', default=100, type=int, help='batch size')
# parser.add_argument('--train_steps', default=1000, type=int,
#                     help='number of training steps')

def gLasso_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # setting the first hidden manually, to add regularizer
    # regularizer = tf.contrib.layers.l1_regularizer(scale=0.6)
    # regularizer = simu_data.group_lasso
    # net = tf.layers.dense(net, units=params['hidden_units'][0],
    #                       activation=tf.nn.relu,
    #                       kernel_regularizer=regularizer,
    #                       name="layer1")
    if params['hidden_units'][0] == 0:
        regularizer = tf.contrib.layers.l1_regularizer(scale=0.6)
        net = tf.layers.dense(net, units=params['hidden_units'][0],
                              activation=tf.nn.relu,
                              kernel_regularizer=regularizer,
                              name="layer1")
    else:
        net = tf.layers.dense(net, units=params['hidden_units'][0],
                              activation=tf.nn.relu)
        weights1 = tf.Variable(tf.truncated_normal([len(features),
                                                    params['hidden_units'][0]]))


    # then for the following layers
    if len(params['hidden_units']) >= 2:
        for units in params['hidden_units'][1:]:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
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

    if params['hidden_units'][0] > 0:
        loss = loss + 0.6 * simu_data.group_lasso(weights1)

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
#     args = parser.parse_args(argv[1:])
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
#             'hidden_units': [3],
#             # The model output.
#             'n_response': 1,
#         })
#
#     # Train the Model.
#     classifier.train(
#         input_fn=lambda: simu_data.train_input_fn(
#             train_x, train_y, args.batch_size),
#         steps=args.train_steps)
#
#     # Evaluate the model.
#     eval_result = classifier.evaluate(
#         input_fn=lambda: simu_data.eval_input_fn(test_x, test_y, args.batch_size))
#
#     # extract variables from model.
#     var_dict = dict()
#     for var_name in classifier.get_variable_names():
#         var_dict[var_name] = classifier.get_variable_value(var_name)
#
#     # feature_weight = classifier.get_variable_value('Variable')
#     # layer1_weight = classifier.get_variable_value('dense/kernel')
#
#     print('\nTest set MSE: {MSE:0.3f}\n'.format(**eval_result))
#     # print('\nWeights of first layer: \n', feature_weight)
#     # print('\nWeights of output layer: \n', layer1_weight)
#     print('\nWeights of output layer: \n', var_dict)
#
#
# if __name__ == '__main__':
#     tf.logging.set_verbosity(tf.logging.INFO)
#     tf.app.run(main)

N_runs = 20
reg_factors = np.e**np.linspace(-10, 5, 16)
n_list = np.array([100, 200, 500, 1000])
method_list = ["linear", "polynomial", "rbf"]
snr_list = np.array([2, 1])
rho_list = np.array([0, .3])
p_list = np.array([20, 50, 100])

csvFile = open("result.csv", "a")
fileHeader = ["Nk", "n", "method", "snr", "rho", "p", "mse"]
writer = csv.writer(csvFile)
writer.writerow(fileHeader)
for k in np.arange(0, N_runs):
    print("Run ", k + 1, " of ", N_runs, "...\n", end="")
    for n in n_list:
        for method in method_list:
            for snr in snr_list:
                for rho in rho_list:
                    for p in p_list:
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
                                'hidden_units': [3],
                                'n_response': 1,
                            })

                        classifier.train(
                            input_fn=lambda: simu_data.train_input_fn(
                                train_x, train_y, 100),
                            steps=1000)

                        eval_result = classifier.evaluate(
                            input_fn=lambda: simu_data.eval_input_fn(test_x, test_y, 100))

                        var_dict = dict()
                        for var_name in classifier.get_variable_names():
                            var_dict[var_name] = classifier.get_variable_value(var_name)

                        # feature_weight = classifier.get_variable_value('Variable')
                        # layer1_weight = classifier.get_variable_value('dense/kernel')

                        print('\nTest set MSE: {MSE:0.3f}\n'.format(**eval_result))
                        # print('\nWeights of first layer: \n', feature_weight)
                        # print('\nWeights of output layer: \n', layer1_weight)
                        print('\nWeights of output layer: \n', var_dict)

                        add_info = [k, n, method, snr, rho, p, eval_result]
                        writer.writerow(add_info)

csvFile.close()
