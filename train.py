import pathlib
import argparse
import os
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from gen_data import LSP_DATA
from model import CPMModel
import matplotlib.pyplot as plt
import seaborn as sns

log_file_path = './my_log.txt'

learning_rate = 1e-4
num_epoch = 100
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

    loss0 = loss(y_true, y_pred[0])
    loss1 = loss(y_true, y_pred[1])
    loss2 = loss(y_true, y_pred[2])
    loss3 = loss(y_true, y_pred[3])
    loss4 = loss(y_true, y_pred[4])
    loss5 = loss(y_true, y_pred[5])

    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
    return total_loss

def train():
    cpm = CPMModel()

    data = LSP_DATA('lspet', training_dataset_path, 8)
    print("---------- Start Training ----------")
    for e in range(num_epoch):
        l = len(data)
        num_error=0

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

                # for j in range(15):
                       
                #     heat = heatmap[:,:, j]

                #     plt.figure()
                #     p1 = sns.heatmap(heat)
                #     figure = p1.get_figure()
                #     figure.savefig("o1" + str(j) + ".png", dpi=400) 

                # heat = image[0,:,:,0]

                # plt.figure()
                # p1 = sns.heatmap(heat)
                # figure = p1.get_figure()
                # figure.savefig("o2" + str(i) + ".png", dpi=400)
                # exit()
                # break
                # exit()             
            except Exception as e:
                # print(e)
                num_error += 1
        # exit()
        print('\nTraining epoch {} with loss {}'.format(e, loss))
        print("num_error:",num_error)
        if e % 1 == 0:
            print('[%05d] Loss: %.4f' % (e, loss))
            with open(log_file_path, "a") as f:
                f.write('\n[%05d] Loss: %.4f' % (e, loss))
            cpm.save_weights(os.path.join("ck",str(e)+" "+str(float(loss))+" loss"))


if __name__ == "__main__":
    train()
