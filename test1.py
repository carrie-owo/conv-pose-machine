import pathlib
import argparse
import os
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from preprocess.gen_data import LSP_DATA
from preprocess.Transformers import Compose, RandomCrop, RandomResized
from model import CPMModel

resume = None
log_file_path = './my_log.txt'
weight_dir = pathlib.Path('weights').absolute()
weights_file = weight_dir.joinpath('cpm.h5')
#Arguments
parser = argparse.ArgumentParser(description='CPM')
parser.add_argument('--resume', default=None, type=str)
args = parser.parse_args()
if args.resume:
    resume = args.resume


learning_rate = 1e-3
num_epoch = 50
batch_size = 1
save_interval = 100
optimizer = keras.optimizers.Adam(learning_rate)
loss = keras.losses.MeanSquaredError()

training_dataset_path = 'lspet_dataset'
val_data_path = 'lsp_dataset'

image_shape = (368,368,3)
centermap_shape = (368,368,1)

def loss_function(y_true, y_pred):

    y_pred = tf.squeeze(y_pred, axis=1)

    # print("y_true.shape: ", y_true.shape)
    # print("y_pred.shape: ", y_pred.shape)

    loss0 = loss(y_true, y_pred[0])
    loss1 = loss(y_true, y_pred[1])
    loss2 = loss(y_true, y_pred[2])
    loss3 = loss(y_true, y_pred[3])
    loss4 = loss(y_true, y_pred[4])
    loss5 = loss(y_true, y_pred[5])

    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
    # print("total_loss: ", total_loss)
    return total_loss

def train():
   
    cpm = CPMModel()

    import skimage.io
    from skimage.io import imread,imshow

    img = imread('a.jpg')
    print("asdfasdfasdfasdf")
    print(img.shape)
    output=cpm(image)
    print(output)
    imshow(output)



    data = LSP_DATA('lspet', training_dataset_path, 8, Compose([RandomResized(), RandomCrop(368)]))
    print("---------- Start Training ----------")
    for e in range(num_epoch):
        l = len(data)
        num_error=0
        # try:
            # for i, d in enumerate(data):
            #     if i%100==0:
            #         print(i,l,i/l)
            #     image, heatmap, centermap = d
            #     image = tf.expand_dims(image, axis=0)
            #     # image = tf.expand_dims(image, axis=0)
            #     centermap = tf.expand_dims(centermap, axis=0)
            #     # centermap = tf.expand_dims(centermap, -1)
            #     print("image.shape: ", image.shape)
            #     print("heatmap.shape: ", heatmap.shape)
            #     print("centermap.shape: ", centermap.shape)
            #     # loss = model.train_on_batch((image, centermap), heatmap)
            #     loss = model.fit((image, centermap), heatmap)
        for i, d in enumerate(data):
            if d is None:
                num_error += 1
                continue
            try:
                if i % 1000 == 0:
                    print(i,l,i/l,num_error)
                image, heatmap, centermap = d
                image = tf.expand_dims(image, axis=0)
                centermap = tf.expand_dims(centermap, axis=0)
                with tf.GradientTape() as tape:
                    output = cpm(image, centermap)
                    loss = loss_function(heatmap, output)
                gradients = tape.gradient(loss, cpm.trainable_variables)
                optimizer.apply_gradients(zip(gradients, cpm.trainable_variables))                   
            except:
                num_error += 1
        # except KeyboardInterrupt:
        #     save_weights()
        #     return    
        print('\nTraining epoch {} with loss {}'.format(e, loss))
        print("num_error:",num_error)
        if e % 1 == 0:
            print('[%05d] Loss: %.4f' % (e, loss))
            with open(log_file_path, "a") as f:
                f.write('\n[%05d] Loss: %.4f' % (e, loss))
            cpm.save_weights(os.path.join("ck",str(e)+" "+str(float(loss))+" loss"))
        # if save_interval and e > 0 and e % save_interval == 0:
        #     save_weights()
        
        # with open(log_file_path, "a") as f:
        #     f.write('\nCurrent time: %s\n' % datetime.now())


if __name__ == "__main__":
    train()
