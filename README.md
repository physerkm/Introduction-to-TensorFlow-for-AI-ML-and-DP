# Introduction to TensorFlow for AI ML and DP

## **An Introduction to computer vision**

In the previous lesson, you learned what the machine learning paradigm is and how you use data and labels and have a computer in fair the rules between them for you. You looked at a very simple example where it figured out the relationship between two sets of numbers. Let's now take this to the next level by solving a real problem, computer vision.

Computer vision is the field of having a computer understand and label what is present in an image. Consider this slide. When you look at it, you can interpret what a shirt is or what a shoe is, but how would you program for that? If an extra terrestrial who had never seen clothing walked into the room with you, how would you explain the shoes to him? It's really difficult, if not impossible to do right? And it's the same problem with computer vision. So one way to solve that is to use lots of pictures of clothing and tell the computer what that's a picture of and then have the computer figure out the patterns that give you the difference between a shoe, and a shirt, and a handbag, and a coat. That's what you're going to learn how to do in this section.

Fortunately, there's a data set called Fashion MNIST which gives a 70,000 images spread across 10 different items of clothing. These images have been scaled down to 28x28 pixels. Now usually, the smaller the better because the computer has less processing to do. But of course, you need to retain enough information to be sure that the features and the object can still be distinguished. If you look at this slide you can still tell the difference between shirts, shoes, and handbags. So this size does seem to be ideal, and it makes it great for training a neural network.

The images are also in gray scale, so the amount of information is also reduced. Each pixel can be represented in values from `0` to `255` and so it's only one byte per pixel. With 28x28 pixels in an image, only 784 bytes are needed to store the entire image. Despite that, we can still see what's in the image and in this case, it's an ankle boot, right?

## **Writing code to load training data**

So what will handling this look like in code? In the previous lesson, you learned about TensorFlow and Keras, and how to define a super simple neural network with them. In this lesson, you're going to use them to go a little deeper but the overall API should look familiar. The one big difference will be in the data. The last time you had your six pairs of numbers, so you could hard code it. This time you have to load 70,000 images off the disk, so there'll be a bit of code to handle that. Fortunately, it's still quite simple because Fashion-MNIST is available as a data set with an API call in TensorFlow.

We simply declare an object of type MNIST loading it from the Keras database. On this object, if we call the load data method, it will return 4 lists to us. That's the training data, the training labels, the testing data, and the testing labels. Now, what are these you might ask? Well, when building a neural network like this, it's a nice strategy to use some of your data to train the neural network and similar data that the model hasn't yet seen to test how good it is at recognizing the images. So in the Fashion-MNIST data set, 60,000 of the 70,000 images are used to train the network, and then 10,000 images, one that it hasn't previously seen, can be used to test just how good or how bad it is performing. So this code will give you those sets.

```
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```


Then, each set has data, the images themselves and labels and that's what the image is actually of. So for example, the training data will contain images like this one, and a label that describes the image like this. While this image is an ankle boot, the label describing it is the number nine.

Why do you think that might be? There's 2 main reasons.

1. Computers do better with numbers than they do with texts.
2. Importantly, this is something that can help us reduce bias. If we labeled it as an ankle boot, we would be biasing towards English speakers. But with it being a numeric label, we can then refer to it in our appropriate language be it English, Chinese, Japanese, or here, even Irish Gaelic.

