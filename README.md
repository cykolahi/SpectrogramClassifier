In creating the dataset for this project, I experiemented with a few different methods:

The first method (failure) involved parsing through the universal_top_songs.csv file trying to find countries associated with songs by finding songs that only appear popular in once country, and also finding the country from which a given song is the most popular in and combining these two lists. This was a failure because the songs arent always associated with their country of origin, and then having to take that data and find corresponding audio files in the FMA archive eliminates a lot of songs and doesn't even capture the songs that are popular in big countries like the US. 


The method that I think will work best is to parse directly throuhg the FMA metadata and extract the audio paths and the corresponding location coordinates. Using the geopy module, I can find the country of the audio file and construct a new csv of audio paths and countries. From there I can balance the number of occurences of each country by splitting up each audio clip into 9 second segments and then creating a spectrogram for each segment. This should give me a dataset of balanced countrys with their corresponding spectrograms. 

The issue that arises with this method is that the FMA-small dataset is heavily unbalanced with songs from the US and UK. This is a problem because the model will be overfit to these countries and not generalize to other countries. To combat this, I can downsample the US, UK and other big countries' songs to be the same number as the least popular country in the dataset. I can then augment the dataset by splitting up the audio clips into 9 second segments. Finally to get the spectrograms of the segmented audio, I use librosa.load to convert the audio clips into tensors and then use librosa.display.specshow to create the spectrograms. 

Rather than using the actual images of the spectrograms, I think it would be better to use the raw tensor data to train the model.  

Running the checkpoint1.ipynb file will run the above methods and save the dataset to a pkl file. 

To see the features of the data run the show_train_test_split.py file. 


