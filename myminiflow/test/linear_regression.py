import myminiflow as tf

def linear_regression(customized_epoch_number=None, verbose=None):
    if customized_epoch_number is None:
        epoch_number = 300
    else:
        epoch_number = customized_epoch_number
    learning_rate = 0.01
    train_features = [1.0, 2.0, 3.0, 4.0, 5.0]
    train_labels = [10.0, 20.0, 30.0, 40.0, 50.0]

    weights = tf.Variable(0.0)
    bias = tf.Variable(0.0)
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    predict = weights * x + bias
    loss = tf.square(y - predict)
    sgd_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = sgd_optimizer.minimize(loss)

    with tf.Session() as sess:
        print(sess)
        sess.run(tf.global_variables_initializer())

        for epoch_index in range(epoch_number):
            sample_number = len(train_features)
            train_feature = train_features[epoch_index % sample_number]
            train_label = train_labels[epoch_index % sample_number]

            sess.run(train_op, feed_dict={x: train_feature, y: train_label})
            loss_value = sess.run(loss, feed_dict={x: 1.0, y: 10.0})

            if verbose:
                print("Epoch: {}, loss: {}, weight: {}, bias: {}".format(
                    epoch_index, loss_value, sess.run(weights), sess.run(bias)))


if __name__ == "__main__":
    linear_regression(customized_epoch_number=1000, verbose=True)

