import numpy as np
import keras

# custom generator for data augmentation
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y=None, batch_size=32, shuffle=True, augment_data=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment_data = augment_data
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X = self.X[indexes]
        if self.y is not None:
            batch_y = self.y[indexes]
            return batch_X, batch_y
        else:
            return batch_X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # traslation
    def translate_keypoints(self, keypoints, max_shift=5):
        shifted_keypoints = keypoints.copy()
        num_keypoints = keypoints.shape[0]
        # calculate random shift for each axis
        shift_x = np.random.randint(-max_shift, max_shift + 1)
        shift_y = np.random.randint(-max_shift, max_shift + 1)
        # apply shift
        shifted_keypoints[:, 0] += shift_x
        shifted_keypoints[:, 1] += shift_y
        return shifted_keypoints

    # vertical flip
    def vertical_flip_keypoints(self, keypoints, height):
        flipped_keypoints = keypoints.copy()
        flipped_keypoints[:, 1] = height - flipped_keypoints[:, 1]
        return flipped_keypoints

    # rotation
    def rotate_keypoints(self, keypoints, angle_range=(-10, 10)):
        rotated_keypoints = keypoints.copy()
        # calculate rotation random angle
        angle = np.random.uniform(angle_range[0], angle_range[1])
        # convert angle to radians
        angle_rad = np.deg2rad(angle)
        # rotation matrix
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        # apply rotation
        rotated_keypoints = np.dot(rotated_keypoints, rotation_matrix)
        return rotated_keypoints

    # scale
    def scale_keypoints(self, keypoints, scale_range=(0.8, 1.2)):
        scaled_keypoints = keypoints.copy()
        # calculate scale random factor
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        # apply scale
        scaled_keypoints *= scale_factor
        return scaled_keypoints

    def __data_augmentation(self, batch_X):
        augmented_batch_X = batch_X.copy()

        for i in range(len(batch_X)):
            frame = batch_X[i]

            # translation
            if np.random.rand() < 0.5:
                frame = self.translate_keypoints(frame, max_shift=5)

            # noise
            if np.random.rand() < 0.5:
                frame = self.add_gaussian_noise(frame, mean=0, std=0.5)

            # horizontal flip
            if np.random.rand() < 0.5:
                frame = self.horizontal_flip_keypoints(frame, width=640)

            # vertical flip
            if np.random.rand() < 0.5:
                frame = self.vertical_flip_keypoints(frame, height=480)

            # rotation
            if np.random.rand() < 0.5:
                frame = self.rotate_keypoints(frame, angle_range=(-10, 10))

            # scale
            if np.random.rand() < 0.5:
                frame = self.scale_keypoints(frame, scale_range=(0.8, 1.2))

            augmented_batch_X[i] = frame

        return augmented_batch_X

    def __data_generation(self, batch_X):
        if self.augment_data:
            batch_X = self.__data_augmentation(batch_X)

        return batch_X
