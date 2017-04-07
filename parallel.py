# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory
import reader
import model
import time
import os
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
list1 = ['cubist','candy','denoised_starry','feathers']

if __name__ == '__main__':
	with tf.Graph().as_default():
		tf.logging.set_verbosity(tf.logging.INFO)
		_, image_preprocessing_fn = preprocessing_factory.get_preprocessing(
					'vgg_16',is_training=False)
		start_time = time.time()
		#width,height = 960,960
		x = tf.placeholder(dtype=tf.string)
		
		image = reader.read_image(x, image_preprocessing_fn)
		image = tf.expand_dims(image, 0)
		generated = model.net(image, training=False)
		generated = tf.squeeze(generated, [0])
		end_time = time.time()
		tf.logging.info('Elapsed time: %fs' % (end_time - start_time))

		sess1 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))		
		saver = tf.train.Saver(tf.global_variables())
		#saver = tf.train.import_meta_graph('models/2/fast-style-model.ckpt-done.meta', clear_devices=True)
		
		sess1.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		
		saver.restore(sess1, 'models/2/fast-style-model.ckpt-done')

		

		sess2 = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		#saver = tf.train.Saver(tf.global_variables())
		#saver = tf.train.import_meta_graph('models/2/fast-style-model.ckpt-done.meta', clear_devices=True)
		sess2.run([tf.global_variables_initializer(), tf.local_variables_initializer()])		
		saver.restore(sess2, 'models/3/fast-style-model.ckpt-done')

		while True:
			y = raw_input('Press o to output:')
			if y == 'o':
				break
			elif y == '0':
				image_dir = 'img/%s.jpg'%list1[0]
				out = sess2.run(generated, feed_dict={x: image_dir})
				out = tf.cast(out, tf.uint8)
				generated_file = 'generated/0.jpg'
				with open(generated_file, 'wb') as img:
					img.write(sess1.run(tf.image.encode_jpeg(out)))
				tf.logging.info('Done. Please check %s.' % generated_file)
			elif y == '1':
				image_dir = 'img/%s.jpg'%list1[1]
				out = sess2.run(generated, feed_dict={x: image_dir})
				out = tf.cast(out, tf.uint8)
				generated_file = 'generated/1.jpg'
				with open(generated_file, 'wb') as img:
					img.write(sess1.run(tf.image.encode_jpeg(out)))
				tf.logging.info('Done. Please check %s.' % generated_file)
			elif y == '2':
				image_dir = 'img/%s.jpg'%list1[2]
				out = sess1.run(generated, feed_dict={x: image_dir})
				out = tf.cast(out, tf.uint8)
				generated_file = 'generated/2.jpg'
				with open(generated_file, 'wb') as img:
					img.write(sess1.run(tf.image.encode_jpeg(out)))
				tf.logging.info('Done. Please check %s.' % generated_file)
			elif y == '3':
				image_dir = 'img/%s.jpg'%list1[3]
				out = sess1.run(generated, feed_dict={x: image_dir})
				out = tf.cast(out, tf.uint8)
				generated_file = 'generated/3.jpg'
				with open(generated_file, 'wb') as img:
					img.write(sess1.run(tf.image.encode_jpeg(out)))
				tf.logging.info('Done. Please check %s.' % generated_file)
			else:
				print ('error key')

