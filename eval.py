# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
#tf.app.flags.DEFINE_string("model_file", "models.ckpt", "")
tf.app.flags.DEFINE_string("image_file", "a.jpg", "")

FLAGS = tf.app.flags.FLAGS
#demo_list = ['0','11','12','15','17','18','2','5','6','7']
#demo_list = ['0_1','11_1','12_1','15_1','16_1','17_1','18_1','2_1','5_1','6_1','7_1',
#    '0_2','11_2','12_2','15_2','16_2','17_2','18_2','2_2','5_2','6_2','7_2','11_5','11_6','11_7']
#demo_list = ['cartoon', 'guohua', 'ice_mountain', 'moon_night', 'sand_painting',
# 'sunset', 'wenli_06', 'shuimo']
demo_list = ['date_0308/sand_painting']

def main(_):
    height = 0
    width = 0
    for head in demo_list:
        model_file = 'models/%s/fast-style-model.ckpt-2000'%head
        with open(FLAGS.image_file, 'rb') as img:
            with tf.Session().as_default() as sess:
                if FLAGS.image_file.lower().endswith('png'):
                    image = sess.run(tf.image.decode_png(img.read()))
                else:
                    image = sess.run(tf.image.decode_jpeg(img.read()))
                height = image.shape[0]
                width = image.shape[1]
        tf.logging.info('Image size: %dx%d' % (width, height))

        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)).as_default() as sess:
                image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
                    FLAGS.loss_model,
                    is_training=False)
                image = reader.get_image(FLAGS.image_file, height, width, image_preprocessing_fn)
                image = tf.expand_dims(image, 0)
                generated = model.net(image, training=False)
                generated = tf.squeeze(generated, [0])
                saver = tf.train.Saver(tf.all_variables())
                sess.run([tf.initialize_all_variables(), tf.initialize_local_variables()])
                #name = FLAGS.model_file.split('/')[1]
                model_file = os.path.abspath(model_file)
                saver.restore(sess, model_file)

                start_time = time.time()
                generated = sess.run(generated)
                generated = tf.cast(generated, tf.uint8)
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
                generated_file = 'generated/result_0338/%s.jpg'%head
                #generated_file = 'generated/%s.jpg'%head
                if os.path.exists('generated') is False:
                    os.makedirs('generated')
                with open(generated_file, 'wb') as img:
                    img.write(sess.run(tf.image.encode_jpeg(generated)))
                    tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
