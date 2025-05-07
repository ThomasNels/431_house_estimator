# 431_house_estimator
This project is for the UNR class CS431, Introduction to Big Data. It uses a zillow data set to predict the accuracy of Zillow's Zestimate feature. 

To run this project you will need either Jupyter Notebook installed, an IDE capable of running Jupyter Notebooks, or load the files into a google colab session. The only file that needs to be run is model_training.ipynb. 

The data files needed to run the models can be obtained at the following: https://www.kaggle.com/c/zillow-prize-1/data 

The last step involved in running the models is to change the following lines in cell 2 of the model_training.ipynb to the location of the previously downloaded files: 
processor = ZillowDataProcessor(
    '/mnt/c/CS431/properties_2016.csv',
    '/mnt/c/CS431/properties_2017.csv',
    '/mnt/c/CS431/train_2016_v2.csv',
    '/mnt/c/CS431/train_2017.csv'
)
