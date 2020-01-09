# Movie Recommendation Neural Network Based on Learned Embeddings From Wikipedia Articles

This is a Tensorflow 2 and Keras neural network project to make movie recommendations for a specified movie using Neural Network Embeddings trained on all articles about movies in the English language Wikipedia. Training the model on this source is predicated on the principle that movies which link to similar Wikipedia pages are therefore similar to one another themselves. To achieve this, this model computes their similarity by learning "embeddings" of movies and their internal Wikipedia links as a 50-dimensional vector.

## How it works

To start with, the `training_data_generator.py` file downloads the current revision of _all Wikipedia articles_ in the latest Wikipedia data dump from [https://dumps.wikimedia.org/enwiki/latest/](https://dumps.wikimedia.org/enwiki/latest/) as compressed partition files. As of the 5th of January 2020, that equates to 59 files, each around 300–400 MB totalling about 16 GB compressed, and expanding to over 60 GB when decompressed! It then processes each of these files, collecting all articles about movies, extracting their titles and links, and finally saving it all as a JSON file called `data/training_data.json`.

> "Embeddings are a way to represent discrete — categorical — variables as continuous vectors. In contrast to an encoding method like one-hot encoding, neural network embeddings are low-dimensional and learned, which means they place similar entities closer to one another in the embedding space."

The input to the model is batches of movie and link pairs (as index integers), where the real (positive) examples are randomly sampled from the training data and have a label of 1, and the false (negative) examples have a label of -1. The output is a prediction of whether or not the link was indeed present in the movie's Wikipedia article. While that is the output of the model, it's just for training; we're not actually interested in the prediction, what we're really after is the learned embeddings. In other words, whereas with most neural networks the weights are how we make the output predictions, in this network it's the other way around; the weights (the embeddings) are the output, and the predictions are how the model learns them.

While training, the model learns similar embeddings for movies that link to similar articles and thus places similar movies next to one another in the embedding space. Once trained, we extract and normalize the embeddings from the embedding layers, calculate the cosine similarity between the embeddings for all movies and the embeddings for the movie that we are querying, and sort them (higher numbers indicate entities that are closer together, with -1 being the furthest apart and 1 being the closest together) to rank each movie by similarity to the query movie (in other words, find the closest neighbours in the embedding space).

This project is adapted from a [notebook](https://github.com/WillKoehrsen/wikipedia-data-science/blob/master/notebooks/Book%20Recommendation%20System.ipynb) by [Will Koehrsen](http://twitter.com/@koehrsen_will).

## Model architecture

__Input layers:__ Two parallel `keras.layers.Input` layers for the movie and link.

__Embedding layers:__ Two parallel 50-dimensional `keras.layers.Embedding` layers, one for each input (movie and link).

__Dot and Reshape layers:__ `keras.layers.Dot` and `keras.layers.Reshape` layers to merge the embedding layers by computing the dot product and shape to a single number.

Please see the code annotations for more details.

## Requirements

Python version: 3.7.4

See dependencies.txt for packages and versions (and below to install).

## Setup

Clone the Git repo.

Install the dependencies:

```bash
pip install -r dependencies.txt
```

## Run

### Download and process training data

__Warning!__ The `download_current_revision_article_partition_files_in_latest_wikipedia_dump()` method in here will download _a lot_ of data; the current revision of literally every article on English language Wikipedia. As of the 5th of January 2020, that was 59 files totalling about 16 GB! Be aware of that when running the below command.

```bash
python -W ignore training_data_generator.py
```

### Build and train the model and output results

```bash
python -W ignore main.py "Blade Runner"
```

If you want to view a horizontal bar chart of the results, set `plot = True` in the call to `find_similar_entities()`.

The above command will output a summary of the model's architecture and the progress of the training, and then print results that will look like the following:

```
Movies most similar to Blade Runner:

Movie: Blade Runner                         Similarity: 1.0
Movie: Blade Runner 2049                    Similarity: 0.98
Movie: Alien                                Similarity: 0.94
Movie: 2001: A Space Odyssey                Similarity: 0.93
Movie: A Clockwork Orange                   Similarity: 0.93
Movie: Jurassic Park                        Similarity: 0.91
Movie: The Shining                          Similarity: 0.90
Movie: The Terminator                       Similarity: 0.88
Movie: The Big Lebowski                     Similarity: 0.88
Movie: Indiana Jones and the Last Crusade   Similarity: 0.87
```

The fact that the queried movie is at the very top of the list with a similarity of exactly 1.0 means that it's working!

## Monitoring/logging

After training, run:

```
$ tensorboard --logdir logs/fit
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.0.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

Then open the above URL in your browser to view the model in TensorBoard.

## Future work

Time permitting, would like to explore making an interactive visualization of the learned embeddings with [TensorFlow’s projector tool](https://projector.tensorflow.org/).
