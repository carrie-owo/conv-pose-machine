import pathlib
import argparse
import signal
from tensorflow import keras
from datetime import datetime

resume = None
log_file_path = './my_log.txt'
weight_dir = pathlib.Path('weights').absolute()
weights_file = weight_dir.joinpath('cpm.h5')

#Arguments
parser = argparse.ArgumentParser(description='U22 Net')
parser.add_argument('--resume', default=None, type=str)
args = parser.parse_args()
if args.resume:
    resume = args.resume


learning_rate = 1e-3
num_epoch = 1e6
save_interval = 100
optimizer = keras.optimizers.Adam(learning_rate)
loss = keras.losses.MeanSquaredError()


def train():
    model = ""
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

    for sig in [signal.SIGABRT, signal.SIGINT, signal.SIGTSTP]:
        signal.signal(sig, autosave)

    print("---------- Start Training ----------")
    for e in range(num_epoch):
        try:
            # inputs, masks = get_training_img_gt_batch(batch_size=batch_size)
            # loss = model.train_on_batch(inputs, masks)
            pass
        except KeyboardInterrupt:
            save_weights()
            return
        except ValueError:
            continue

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