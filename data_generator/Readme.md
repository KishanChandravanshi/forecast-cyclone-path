# Information:
when you will run the augmented_data.py all the data present in the dataset folder will be copied and augmented and will be save in augmented_data folder. You then need to copy that data to dataset folder, to extend your dataset.
<br>
If your dataset size is larger, then run resize_images.py, it will take the data present in your dataset folder and resize it to (500 x 500) and will put it in resized_data folder. You need to first delete the data present in the dataset folder and replace it with this new data if required.
<br>
After going through the preprocessing phase, we need to run draw_box.py