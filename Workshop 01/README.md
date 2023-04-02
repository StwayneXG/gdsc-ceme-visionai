# Workshop 1 - Sign Language Classification

## Introduction

In this workshop, we will be building a sign language classification model using Convolutional Neural Networks (CNNs). The goal of the workshop is to introduce you to the basics of CNNs and how they can be used to solve real-world problems. Moreover, our objective is to familiarize you with the machine learning pipeline and the tools used in the process.

## Dataset

We will be using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to train our model. The dataset contains 26 different letters of the English alphabet. The images are 28x28 pixels in size and are in grayscale. 

The dataset is split into:

- 60,000 training images 
- 10,000 test images.

Note: There are no cases for J or Z because of gesture motions

You can find the dataset in the `dataset` folder.

## Technical Details

### Model

We will be using a CNN to classify the images. The model consists of 3 convolutional layers with batch normalization and 2 fully connected layers. The model architecture is as follows:

```python
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(26, activation='softmax'))
```

### Training

We will be using the RMSprop optimizer to train the model. The model will be trained for 10 epochs with a batch size of 64. The loss function used is categorical cross-entropy.

```python
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size = 64, epochs = 10, validation_data = (X_test, Y_test))
```

### Evaluation

The model is evaluated on the test set. The accuracy achieved is 99.2%.

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
```

### Results

The confusion matrix is as follows:

## References

- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Jupyter Notebook](https://jupyter.org/)
- [Google Colab](https://colab.research.google.com/)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
