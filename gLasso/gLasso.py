
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import simu_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def gLasso_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # setting the first hidden manually, to add regularizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    net = tf.layers.dense(net, units=params['hidden_units'][0],
                          activation=tf.nn.relu,
                          kernel_regularizer=regularizer,
                          name="layer1")

    # then for the following layers
    if len(params['hidden_units']) >= 2:
        for units in params['hidden_units'][1:]:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    response = tf.layers.dense(net, params['n_response'], activation=None)

    # Compute predictions.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "response": response,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.mean_squared_error(labels=tf.case(tf.expand_dims(labels, axis=1), tf.float32),
                                        predictions=response)

    # Compute evaluation metrics.
    mse = tf.metrics.mean_squared_error(labels=tf.case(tf.expand_dims(labels, axis=1), tf.float32),
                                        predictions=response)
    metrics = {'MSE': mse}
    tf.summary.scalar("MSE", mse)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    layer1_weight = tf.trainable_variables("layer1")
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = simu_data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.Estimator(
        model_fn=gLasso_model,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 20 nodes each.
            'hidden_units': [20, 20],
            # The model output.
            'n_response': 1,
        })

    # Train the Model.
    classifier.train(
        input_fn=lambda:simu_data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:simu_data.eval_input_fn(test_x, test_y, args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

