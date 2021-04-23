import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
class CPMModel(tf.keras.models.Model):
    def __init__(self):
        super(CPMModel, self).__init__()
        self.num_pose = 14
        self.pool_center = MaxPool2D(9,8, padding="same")

        self.conv1_stage1 = Conv2D(128, 9, padding="same",activation="relu")
        self.pool1_stage1 = MaxPool2D(3,2, padding="same")
        self.conv2_stage1 = Conv2D(128, 9, padding="same",activation="relu")
        self.pool2_stage1 = MaxPool2D(3,2, padding="same")
        self.conv3_stage1 = Conv2D(128, 9, padding="same",activation="relu")
        self.pool3_stage1 = MaxPool2D(3,2, padding="same")
        self.conv4_stage1 = Conv2D(32, 5, padding="same",activation="relu")
        self.conv5_stage1 = Conv2D(512, 9, padding="same",activation="relu")
        self.conv6_stage1 = Conv2D(512, 1, padding="same",activation="relu")
        self.conv7_stage1 = Conv2D(self.num_pose+1, 1, padding="same")

        self.conv1_stage2 = Conv2D(128, 9, padding="same",activation="relu")
        self.pool1_stage2 = MaxPool2D(3,2, padding="same")
        self.conv2_stage2 = Conv2D(128, 9, padding="same",activation="relu")
        self.pool2_stage2 = MaxPool2D(3,2, padding="same")
        self.conv3_stage2 = Conv2D(128, 9, padding="same",activation="relu")
        self.pool3_stage2 = MaxPool2D(3,2, padding="same")

        self.conv4_stage2 = Conv2D(32, 5, padding="same",activation="relu")
        self.Mconv1_stage2 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv2_stage2 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv3_stage2 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv4_stage2 = Conv2D(128, 1, padding="same",activation="relu")
        self.Mconv5_stage2 = Conv2D(self.num_pose+1, 1, padding="same")

        self.conv1_stage3 = Conv2D(32, 5, padding="same",activation="relu")
        self.Mconv1_stage3 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv2_stage3 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv3_stage3 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv4_stage3 = Conv2D(128, 1, padding="same",activation="relu")
        self.Mconv5_stage3 = Conv2D(self.num_pose+1, 1, padding="same")

        self.conv1_stage4 = Conv2D(32, 5, padding="same",activation="relu")
        self.Mconv1_stage4 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv2_stage4 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv3_stage4 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv4_stage4 = Conv2D(128, 1, padding="same",activation="relu")
        self.Mconv5_stage4 = Conv2D(self.num_pose+1, 1, padding="same")

        self.conv1_stage5 = Conv2D(32, 5, padding="same",activation="relu")
        self.Mconv1_stage5 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv2_stage5 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv3_stage5 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv4_stage5 = Conv2D(128, 1, padding="same",activation="relu")
        self.Mconv5_stage5 = Conv2D(self.num_pose+1, 1, padding="same")

        self.conv1_stage6 = Conv2D(32, 5, padding="same",activation="relu")
        self.Mconv1_stage6 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv2_stage6 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv3_stage6 = Conv2D(128, 11, padding="same",activation="relu")
        self.Mconv4_stage6 = Conv2D(128, 1, padding="same",activation="relu")
        self.Mconv5_stage6 = Conv2D(self.num_pose+1, 1, padding="same")

    def stage1(self, image):
        x = self.pool1_stage1(self.conv1_stage1(image))
        x = self.pool2_stage1(self.conv2_stage1(x))
        x = self.pool3_stage1(self.conv3_stage1(x))
        x = self.conv4_stage1(x)
        x = self.conv5_stage1(x)
        x = self.conv6_stage1(x)
        x = self.conv7_stage1(x)
        return x

    def middle(self, image):
        x = self.pool1_stage2(self.conv1_stage2(image))
        x = self.pool2_stage2(self.conv2_stage2(x))
        x = self.pool3_stage2(self.conv3_stage2(x))
        return x

    def stage2(self, pool3_stage2_map, conv7_stage1_map, pool_center_map):
        x = self.conv4_stage2(pool3_stage2_map)
        x = tf.concat([x, conv7_stage1_map, pool_center_map], axis=-1)
        x = self.Mconv1_stage2(x)
        x = self.Mconv2_stage2(x)
        x = self.Mconv3_stage2(x)
        x = self.Mconv4_stage2(x)
        x = self.Mconv5_stage2(x)
        return x

    def stage3(self, pool3_stage2_map, Mconv5_stage2_map, pool_center_map):
        x = self.conv1_stage3(pool3_stage2_map)
        x = tf.concat([x, Mconv5_stage2_map, pool_center_map], axis=-1)
        x = self.Mconv1_stage3(x)
        x = self.Mconv2_stage3(x)
        x = self.Mconv3_stage3(x)
        x = self.Mconv4_stage3(x)
        x = self.Mconv5_stage3(x)
        return x

    def stage4(self, pool3_stage2_map, Mconv5_stage3_map, pool_center_map):
        x = self.conv1_stage4(pool3_stage2_map)
        x = tf.concat([x, Mconv5_stage3_map, pool_center_map], axis=-1)
        x = self.Mconv1_stage4(x)
        x = self.Mconv2_stage4(x)
        x = self.Mconv3_stage4(x)
        x = self.Mconv4_stage4(x)
        x = self.Mconv5_stage4(x)
        return x

    def stage5(self, pool3_stage2_map, Mconv5_stage4_map, pool_center_map):
        x = self.conv1_stage5(pool3_stage2_map)
        x = tf.concat([x, Mconv5_stage4_map, pool_center_map], axis=-1)
        x = self.Mconv1_stage5(x)
        x = self.Mconv2_stage5(x)
        x = self.Mconv3_stage5(x)
        x = self.Mconv4_stage5(x)
        x = self.Mconv5_stage5(x)
        return x

    def stage6(self, pool3_stage2_map, Mconv5_stage5_map, pool_center_map):
        x = self.conv1_stage6(pool3_stage2_map)
        x = tf.concat([x, Mconv5_stage5_map, pool_center_map], axis=-1)
        x = self.Mconv1_stage6(x)
        x = self.Mconv2_stage6(x)
        x = self.Mconv3_stage6(x)
        x = self.Mconv4_stage6(x)
        x = self.Mconv5_stage6(x)
        return x

    def call(self, image, center_map):
        
        pool_center_map = self.pool_center(center_map)
        conv7_stage1_map = self.stage1(image)
        pool3_stage2_map = self.middle(image)
        Mconv5_stage2_map = self.stage2(pool3_stage2_map, conv7_stage1_map, pool_center_map)
        Mconv5_stage3_map = self.stage3(pool3_stage2_map, Mconv5_stage2_map, pool_center_map)
        Mconv5_stage4_map = self.stage4(pool3_stage2_map, Mconv5_stage3_map, pool_center_map)
        Mconv5_stage5_map = self.stage5(pool3_stage2_map, Mconv5_stage4_map, pool_center_map)
        Mconv5_stage6_map = self.stage6(pool3_stage2_map, Mconv5_stage5_map, pool_center_map)

        output = tf.stack([conv7_stage1_map, Mconv5_stage2_map, Mconv5_stage3_map, Mconv5_stage4_map, Mconv5_stage5_map, Mconv5_stage6_map])
        return output

if __name__ == "__main__":
    import numpy as np
    model=CPMModel()
    input=np.ones([1,368,368,3],dtype=np.float64)
    centermap = np.zeros((1,368, 368, 1), dtype=np.float32)
    output=model(input,centermap)
    print(output.shape) # (6, 1, 46, 46, 15)
    
