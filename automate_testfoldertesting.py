import tensorflow as tf, sys
import glob
import shutil
import os

image_dir = sys.argv[1]

label_lines = [line.rstrip() for line
               in tf.gfile.GFile("/tf_files/retrained_labels_3eye.txt")]

with tf.gfile.FastGFile("/tf_files/retrained_graph_3eye.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            print " "
            
            ext = file
            ext = os.path.splitext(ext)[-1].lower()
            if(ext == ".jpg"):
                image_path = os.path.join(root, file)
                image_data = tf.gfile.FastGFile(image_path, 'rb').read()
                predictions = sess.run(softmax_tensor, \
                                       {'DecodeJpeg/contents:0': image_data})
                    
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                for node_id in top_k:
                    human_string = label_lines[node_id]
                    score = predictions[0][node_id]
                    print('%s (score = %.5f)' % (human_string, score))
                    break

print "End"
