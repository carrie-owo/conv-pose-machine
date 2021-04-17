import pathlib
import argparse
# import signal
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
num_epoch = 1000000
save_interval = 100
optimizer = keras.optimizers.Adam(learning_rate)
loss = keras.losses.MeanSquaredError()

training_dataset_path = 'lspet_dataset'
val_data_path = 'lsp_dataset'

image_shape = (1,368,368,3)
centermap_shape = (1,368,368,1)

def loss_function(y_true, y_pred):
    loss0 = loss(y_true, y_pred[0])
    loss1 = loss(y_true, y_pred[1])
    loss2 = loss(y_true, y_pred[2])
    loss3 = loss(y_true, y_pred[3])
    loss4 = loss(y_true, y_pred[4])
    loss5 = loss(y_true, y_pred[5])

    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5
    return total_loss

def train():
    image_input = keras.Input(shape=image_shape)
    centermap_input = keras.Input(shape=centermap_shape)
    cpm = CPMModel()
    outputs = cpm(image_input, centermap_input)
    model = keras.Model(inputs=[image_input, centermap_input], outputs=outputs, name='CPMModel')
    model.compile(optimizer=optimizer, loss=loss_function, metrics=None)
    model.summary()

    with open(log_file_path, "a") as f:
        f.write('\nTraining begins at: %s\n' % datetime.now())

    if resume:
        print('Loading weights from %s' % resume)
        model.load_weights(resume)

    # helper function to save state of model
    def save_weights():
        print('Saving state of model to %s' % weights_file)
        with open(log_file_path, "a") as f:
            f.write('\nSaving state of model to %s' % weights_file)
        model.save_weights(str(weights_file))

    # signal handler for early abortion to autosave model state
    def autosave(sig, frame):
        print('Training aborted. Saving weights.')
        save_weights()
        exit(0)

    # for sig in [signal.SIGABRT, signal.SIGINT, signal.SIGTSTP]:
    #     signal.signal(sig, autosave)

    data = LSP_DATA('lspet', training_dataset_path, 8, Compose([RandomResized(), RandomCrop(368)]))
    print("---------- Start Training ----------")
    for e in range(num_epoch):
        l=len(data)
        try:
            for i, d in enumerate(data):
                if i%100==0:
                    print(i,l,i/l)
                image, heatmap, centermap = d
                image = tf.expand_dims(image, axis=0)
                image = tf.expand_dims(image, axis=0)
                centermap = tf.expand_dims(centermap, axis=0)
                centermap = tf.expand_dims(centermap, -1)
                loss = model.train_on_batch((image, centermap), heatmap)
        except KeyboardInterrupt:
            save_weights()
            return    
        print('\nTraining epoch {} with loss {}'.format(e, loss))
        if e % 10 == 0:
            print('[%05d] Loss: %.4f' % (e, loss))
            with open(log_file_path, "a") as f:
                f.write('\n[%05d] Loss: %.4f' % (e, loss))

        if save_interval and e > 0 and e % save_interval == 0:
            save_weights()
        
        with open(log_file_path, "a") as f:
            f.write('\nCurrent time: %s\n' % datetime.now())


if __name__ == "__main__":
    train()
