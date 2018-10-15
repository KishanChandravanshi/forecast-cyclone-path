# forecast-cyclone-path

## About
Cyclone is a system of winds that rotates about a centre of low atmospheric pressure. The model presented in this repository tracks the eye (or the centre) of the cyclone. The goal of the model is to analyse the pattern in its movement and using that to predict its path.
By knowing a general approximation of where the cyclone can hit, will then allow us to warn the people, so that the place could be put on a warning stage, or may be evacuated beforehand which will lead to fewer losses of lives.
As we know that the techniques of deep learning are exploiting a lot in various fields we decided to use it in our model.

## How did we got the Dataset?
To be honest, we didn't had any annotated dataset, though we got the satellite images of the cyclone over the internet, some of them were random and other we got from Indian meteorological department, which had a lot of archived satellite images of cyclones that had hit India in the past.
The data we were able to collect was not enough to create a robust model, so we used data augmentation techniques, in which we rotated the images at different angles, thus changing the position of our cyclone(eye).

## How did we annotated the data?
Well, there are a lot of websites in which you can manually annotate the images, but pretty much all of them are paid we guess. As we only need to label only one class i.e. the eye of the cyclone, we used Matplotlib an important plotting library in python to rescue us from this situation.
Matplotlib has a lot of inbuilt functions that can be used to achieve the task, as we only needed to draw the bounding box across the eye, we only need two points to define a rectangle right?. So we used event detector which captures the coordinates of the location where we pressed the click, so we created a box with that click and when at the time we leave the click, it captures it too. So, in total we got two locations that can be used to define a rectangle.
We then stored these information in a XML file format. which we will discuss later.

## What algorithm did we used for detection and localisation of the eye?
We used, You only look once (YOLO) a state-of-the-art, real-time object detection system. To be more precise we used YOLOv2 architecture. Whose detail you can find here https://pjreddie.com/darknet/yolov2/. It consists of 23 convolutional layers and it uses batch normalization technique and leaky Rectified linear unit (ReLu) activation. The model was trained over 700 images for 1000 epochs on a GTX 1080, 8 GB VRAM, after which the loss started to saturates.

## How did we predicted the path of the cyclone?
In order to test the model we actually need to know the complete path beforehand. So here we used the complete dataset of one cyclone event named <b> HUDHUD </b>. We had its images taken every half an hour. In total we had approximately 4-5 days data. So roughly we had about 120 images in which some were not looking like a cyclone at all, so we have to manually remove them. We then sequentially inputted each image to the previous created model to predict the bounding box on the eye, or to localise the eye. Doing so we extracted the midpoint of the bounding boxes, i.e the x-coordinate and y-coordinate in the csv format in order. So in total we had 120 x and y coordinate. Now using these coordinate we tried to predict the next x, y coordinate.
This was achieved by using an RNN model comprising of LSTM, in which each LSTM layer consists of 100 nodes, followed by a dropout layer set to 50%, this dropout layer helped the model from overfitting. We used in total five LSTM layers and five dropout layers, the output layer was a dense layer having a single output that gave us our coordinate. So in total we used two RNN model one for x-coordinate and other for y-coordinate.
The RNN model was not trained over all the csv, rather we only trained our model on first 100 (x, y) coordinates and then tested it, on how accurately it is able to predict the rest 20 (x, y) coordinates.

## Block diagram representation of the model
 ![Block Diagram](https://github.com/KishanChandravanshi/forecast-cyclone-path/blob/master/tmp/block.png)

## Prediction

### x - coordinate
 ![x-coordinate prediction](https://github.com/KishanChandravanshi/forecast-cyclone-path/blob/master/tmp/x_coordinate.png)
### y - coordinate
 ![x-coordinate prediction](https://github.com/KishanChandravanshi/forecast-cyclone-path/blob/master/tmp/y_coordinate.png)

## To get started follow the steps
* let's clone yolo repository, to do so type the following commands in your terminal or cmd
```
git clone https://github.com/thtrieu/darkflow.git
cd darkflow
```
* Now before moving forward we need to setup a few thing so that the YOLO works properly<br>
`python3 setup.py build_ext --inplace`
<br> Note: if you get an error, <b>'python3 is not recognized'</b>, type python instead of python3 <br>
* Now clone this repo, but in a different location, after doing that copy all the files present in this repo to the previous repository i.e inside darkflow repo.
* You've completed the first phase, now its time to annotate the images, if you already have annotated images, then move to step 4,
* To annotate your image, first move to the data_generator folder, there you will find draw_boxes.py python file, before running it make sure you put all your images in dataset folder, then running draw_boxes.py. You'll be presented with an image, draw the bounding box and when you are done, hit q, to annotate the next image.
* After annotating your image its time to train the model, but before doing so we need to make some adjustment to certain files in order to make them fit for this training.
* Download these files from this link <i>https://drive.google.com/open?id=1NFnCjMnILvQsXYDj2s7rpjckGjZu77ML</i>
* Paste the <b><i>yolo-v2.cfg</i></b> in the <b>cfg</b> directory and <b><i>yolo.weights</i></b> in <b>bin</b> directory.
* Now open your CMD or terminal and paste the following code<br>
`python flow --model cfg/yolo-v2.cfg --load bin/yolo.weights --train --annotation data_generator/annotations --dataset data_generator/dataset --gpu 0.8 --epoch 50`
<br> <b> Note: if you don't have NVIDIA gpu then don't write --gpu 0.8 </b>

* You've completed the second phase, now let's move to the next phase, i.e creation of CSV files and training your RNN
* run the generate_csv.py file, it will basically give you a csvfile.txt as an output, use external software such as MATLAB to convert it into a csvfile and name it 'final.csv'
* run the <b>rnn.py</b> to train your RNN model, it will give you a graph of the predicted result, i.e x-coordinates and y-coordinates.
