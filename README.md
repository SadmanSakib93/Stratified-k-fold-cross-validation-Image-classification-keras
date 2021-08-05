# Image Classification using Stratified-k-fold-cross-validation
This python program demonstrates image classification with stratified k-fold cross validation technique. Libraries required are keras, sklearn and tensorflow.
* The DS.zip file contains a sample dataset that I have collected from Kaggle. It consists of three folders (Train, Test, and Validation) and each of these three folders consists of folders according to class labels (e.g., circles, triangles, and squares).
* Checkout the code in **stratified_K_fold_CV.ipynb** notebook.
* Change the batch size, epoch, and the structure of the CNN model according to your needs. Here, some naive values are provided without any hyper-parameter tuning.
* Change the contents of DS folder according to your dataset/images. But initially keep all the images inside the sub-folders of the "train" folder. The program will take care of splitting the test images and validation images inside the code. Also, don't forget to rename your sub-folders inside train, test and validation folder according to your classes.
 
* Please see the following paper where I used a similar approach:
https://www.researchgate.net/publication/344343833_DL-CRC_Deep_Learning-based_Chest_Radiograph_Classification_for_COVID-19_Detection_A_Novel_Approach
