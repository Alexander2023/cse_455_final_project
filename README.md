# cse_455_final_project

Problem Description:
-
Determine the correct facial expression of an individual in an input image.

Previous Work:
-
* Pretrained model (ResNet)
* Dataset (FER 2013)
* Parts of learning process (cited in code)

Implemented For Project:
-
* Random horizontal flip
* Random rotation using Three Shear Rotation algorithm (described here http://datagenetics.com/blog/august32013/index.html)
* K-Fold Cross-Validation procedure (described here https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right)
* Remaining parts of learning process
* Mini-app for expression recognition and image capture

My Approach:
-
Train a pretrained ResNet model on the FER 2013 dataset while applying data augmentation techniques (random horizontal flip, random rotation). Use Hold-Out method to get an idea of model and configuration performance when exploring options and then using K-Fold Cross-Validation to more rigorously compare top candidate models and configurations. Lastly, using a mini-app to test the model in real-time on recognizing expressions and capturing happy images.

Datasets:
-
* FER 2013 - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

Note: must download and place the train.csv and test.csv in a folder titled fer2013 to reproduce

Results:
-
Note: Top performing models used the model and config in the most recent code

<u>Using Hold-Out method with 80-20 training-testing split</u>

Without data augmentation:

Training accuracy (each epoch): 0.5300313370473537, 0.651375348189415, 0.7104369777158774, 0.7604456824512534 </br>
Testing accuracy: 0.6288194444444445

With data augmentation:

Training accuracy (each epoch): 0.5151027158774373, 0.6207782033426184, 0.6571204735376045, 0.6895020891364902 </br>
Testing accuracy: 0.6385416666666667

<u>Using K-Fold Cross-Validation with 5 folds</u>

Without data augmentation:

Average validation accuracy = 0.6326041666666666

With data augmentation:

Average validation accuracy = 0.6261458333333334

<u>Explanation</u>

From the experimental data above, it appears that data augmentation helps reduce overfitting since training accuracy, through its successive epochs, ends up closer to the testing accuracy when using data augmentation compared to without. However, to my surprise, the validation accuracies from K-Fold Cross-Validation are very close, with the learning process without data augmentation doing slightly better. This may be due to the K folds parameter not being high enough to give adequate results or the data augmentation happening too rarely (probabilities were set to 50% for flipping and 20% for rotating to reduce training and testing time).

Discussion:
-

<u>Problems encountered</u>

An initial problem I encountered was slow training when on a CPU, which made it difficult to tune the hyperparameters. Once I switched to using a GPU on Google Colab, training speed improved dramatically. Another problem I encountered was gaps in images when performing the three shear rotation algorithm. This confused me for a while since it was specifically intended to prevent this compared to using the basic trigonometric rotation. However, I eventually had the idea to round the coordinates after each shear, since each shear transformation can represent new pixel coordinates when applied alone, which ended up fixing the problem. Another problem I encountered was slower training when using data augmentation.

<u>Next steps</u>

One next step for the project if I kept working on it would be to train on larger models. Since I was time-constrained, I was not able to train extensively. Another next step would be to gather new data to make the model generalize better. I would also pre-process the images with data augmentation prior to training to avoid slowdown.

<u>Approach uniqueness</u>

My approach differs from others by applying K-Fold Cross-Validation to top performing models rather than relying on the Hold-Out method throughout all comparisons. This was beneficial since the results from using Hold-Out made it look like the config with data augmentation outperformed that without data augmentation. However, with K-Fold Cross-Validation, I was able to obtain unexpected results which made me think more about whether it was beneficial or if future experimentation would be needed. My approach also differs in that I used real-time video footage to test the model's performance with my mini-app. This was beneficial since I was able to better determine how well the model generalized by applying it to a new environment.

Video:
-
