# Captcha-hacker
hack captcha with CNN

This project is aimed at building a captcha hacker with CNN. I'm new to CNN, so it's also a project to play for fun and knowledge.

The idea behind this model is from "Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks", a paper published by Google's reCAPTCHA Teams.

In my opinion, the key idea of this paper are:

(a) we don't need to split the task into three parts(i.e., localization, segmentation and recognition), it's all done with CNN.

(b) To deal with changeable length of the sequence, we need to add a new output to predict the length of the sequence.

The paper has achieved 99.8% accuracy, hope it will also work for me.

I'm going to split the project into three parts:

(a) First, due to the lack of the labeled data, I need to write my own generator of captcha, it should cover as many types of captcha as possible, so I need to make it flexible to add new features easily.

(b) I'm going to implenment CNN with keras(a friendly library to build neural networks).

(c) Tune CNN to get good performence.

Ok, let's get started!

**UPDATE (2016.8.17)**

This project is no longer maintained, please check the new version of it,[my new project: DeepLearning-OCR](https://github.com/xingjian-f/DeepLearning-OCR). The code is much more optimized, and it has been used in our inner crawler and OCR project.
