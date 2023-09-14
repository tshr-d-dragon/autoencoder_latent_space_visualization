from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE


# Load the train and test data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("X_train : ", X_train.shape, "X_test : ", X_test.shape)

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255


# Define the Autoencoder Model
input_img = Input(shape=(28, 28, 1))

x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Flatten()(x)
x = Dense(128)(x)     # at this point the representation is 128-dimensional
x = BatchNormalization()(x)
x = Activation('elu')(x)

x = Dense(512)(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)

x = Reshape((4, 4, 32))(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3))(x)
x = BatchNormalization()(x)
x = Activation('elu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)


# Visualization of the Autoencoder model
autoencoder.summary()
autoencoder_img_file = 'autoencoder_model.png'
tensorflow.keras.utils.plot_model(autoencoder, to_file=autoencoder_img_file, show_shapes=True)


# Training the Autoencoder model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = autoencoder.fit(X_train, X_train,
                        epochs=100,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        callbacks = EarlyStopping(monitor="val_loss", patience=5))


# Visualization history of trained model
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()


# Visualization of the predictions on the test dataset
decoded_imgs = autoencoder.predict(X_test[:10000])
n = 10  
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# Saving the model
autoencoder.save('autoencoder.h5')


# Exracting the latent space vectors from the test dataset using th bottleneck layer of the Autoencoder model
latent_space = autoencoder.get_layer('activation_3').output
latent_space_model = Model(input_img, latent_space)
# latent_space_model.save('latent_space_model.h5')
latent_space_data = latent_space_model.predict(X_test[:10000]).reshape(10000, 128)
print(latent_space_data.shape, latent_space_data[0].shape)


# Visualization of the raw images flattened and then reduced from 784 to 2 using tSNE algorithm
def getImage(img):
   return OffsetImage(img, zoom=0.1)

tsne_original = TSNE(n_components=2, init="pca", random_state=0, perplexity=50)
trans_tsne_original = tsne_original.fit_transform(X_test[:10000].reshape(10000, 784))

plt.figure(figsize = (30, 15), dpi = 300) 
fig, ax = plt.subplots()

c = 0
for x0, y0 in zip(trans_tsne_original[:,0], trans_tsne_original[:,1]):
   ab = AnnotationBbox(getImage(X_test[c]), (x0, y0), frameon=False)
   c += 1
   ax.add_artist(ab)

scatter = plt.scatter(trans_tsne_original[:,0], trans_tsne_original[:,1], edgecolors = 'none', c = y_test, cmap = 'Paired')
plt.legend(handles=scatter.legend_elements()[0], labels = list(map(str, np.unique(y_test))), bbox_to_anchor = (1.15, 0.8), fontsize = 'small')
plt.savefig('img_784_tsne_perplexity_50.png', dpi = 300)
plt.show()


# Visualization of the raw images flattened and then reduced from 128 to 2 using tSNE algorithm
tsne_latent_space = TSNE(n_components=2, init="pca", random_state=0, perplexity=50)
trans_tsne_latent_space = tsne_latent_space.fit_transform(latent_space_data.reshape(10000, 128))

plt.figure(figsize = (30, 15), dpi = 300) 
fig, ax = plt.subplots()


c = 0
for x0, y0 in zip(trans_tsne_latent_space[:,0], trans_tsne_latent_space[:,1]):
   ab = AnnotationBbox(getImage(X_test[c]), (x0, y0), frameon=False)
   c += 1
   ax.add_artist(ab)

scatter = plt.scatter(trans_tsne_latent_space[:,0], trans_tsne_latent_space[:,1], edgecolors = 'none', c = y_test[:10000], cmap = 'Paired')
plt.legend(handles=scatter.legend_elements()[0], labels = list(map(str, np.unique(y_test))), bbox_to_anchor = (1.15, 0.8), fontsize = 'small')
plt.savefig('img_128_tsne_perplexity_50.png', dpi = 300)
plt.show()