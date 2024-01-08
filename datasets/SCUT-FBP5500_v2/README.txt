README
--------------------------
(1) Folder "Images" contains 5500 frontal, unoccluded faces aged from 15 to 60 with neutral expression. It can be divided into four subsets with different races and gender, including 2000 Asian females, 2000 Asian males, 750 Caucasian females and 750 Caucasian males.

(2) Sheet¡°All_Ratings.xlsx¡±includes the ratings of 60 volunteers. All the images are labeled with beauty scores ranging from [1, 5] by totally 60 volunteers. "original Rating" representa the previous rating of some raters who was randomly equired to rate some of the images again.

(3) Folder "train_test_files" contains the training and testing files. There are two kinds of validation ways, including: 1)5-folds cross validations (for each validation, 80% samples are used for training and the rest for testing); 2)the split of 60% training and 40% testing (60% samples are used for training and the rest for testing). 

(4) Folder "facial landmark" contains the landmark files of 5500 faces with "pts" format. We artificially labeled 86 coordinates for each face.  

(5) Sheet "Images_Sources.xlsx" includes the sources of each facial images. Most of the images were collected from Internet, where some portion of Asian faces were from the DataTang and GuangZhouXiangSu, and some Caucasian faces were from the 10k US Adults Face database. 

(6) Please consider to cite our paper when you use our database:

@article{liang2017SCUT,
title     = {SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction},
author    = {Liang, Lingyu and Lin, Luojun and Jin, Lianwen and Xie, Duorui and Li, Mengru},
jurnal    = {ICPR}, 
year      = {2018}
}

(7) For any questions about this database, please contact the authors by sending email to lianwen.jin@gmail.com and lianglysky@gmail.com

