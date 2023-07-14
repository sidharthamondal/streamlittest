# Object Classification

This is a Python project that uses a convolutional neural network (CNN) to classify objects in images using the CIFAR-10 dataset. The project consists of two main parts: training the model and performing predictions on uploaded images.

## Training the Model

The `train_model.py` file contains the code for training the CNN model. Here are the steps involved:

1. Load the CIFAR-10 dataset from the local directory.
2. Preprocess the training and test data by reshaping the images, normalizing pixel values, and converting labels to one-hot encoded vectors.
3. Define the architecture of the model using convolutional and dense layers with Leaky ReLU activation.
4. Compile the model with the Adam optimizer and categorical cross-entropy loss.
5. Train the model on the training data for a specified number of epochs and batch size.
6. Evaluate the trained model on the test data and print the test loss and accuracy.
7. Save the trained model to a file named `classification_model.h5`.

## Performing Predictions

The `app.py` file contains the code for a Streamlit web application that allows users to upload an image and get a prediction of the object in the image. Here are the main functionalities:

1. Load the saved model from the `classification_model.h5` file.
2. Define the labels for prediction, representing different object classes.
3. Configure the Streamlit layout and add custom CSS for appearance enhancement.
4. Create a two-column layout with the left panel displaying the model capabilities and the right panel for image upload and prediction.
5. In the left panel, display the list of object labels that the model can predict.
6. In the right panel, provide an option to upload an image file (supports JPG, JPEG, and PNG formats).
7. Perform prediction on the uploaded image using the trained model.
8. Display the predicted label along with the uploaded image.

## How to use ##

## 1.  Home Page ##

![A screenshot of a computer Description automatically
generated](/Images/01.png)

This is the home page of our application where it indicates our
application name "Object Classification".

Our Model has the capabilities to predict the \"airplane\",
\"automobile\", \"bird\", \"cat\", \"deer\", \"dog\", \"frog\",
\"horse\", \"ship\", \"truck\".

As we can see that there is a dialogue box with a browse button from
where we can upload an image and image size can be up to 200Mb per file.

## 2. Image Upload : ##

![A screenshot of a computer Description automatically
generated](/Images/02.png)

As we can see once we click on the browse button the Local Files Windows
pops up from where we can upload an image and check the object for the
classification process.

## 3. Final Output : ##

![A cat sitting on a ledge Description automatically
generated](/Images/03.png)

This is the page where the output of the image given to application is
shown.

As we can see we upload the cat images the application give the result
into predicted label: cat.

So, this how our application works for the object classification.
