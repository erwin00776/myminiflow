import myminiflow as tf

if __name__ == '__main__':
    a = tf.constant(10)
    b = tf.constant(20)
    c = a + b
    sess = tf.Session()
    print(sess.run(c))

    d = tf.constant(50)
    e = a + b * d
    print(sess.run(e))

    x = tf.placeholder(dtype=tf.float32, name="x")
    z = tf.add(a, x)
    print(sess.run(z, feed_dict={x: 11}))
