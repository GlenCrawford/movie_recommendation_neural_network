import json
import functools
import operator
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
import training_data_generator

random.seed(100)

# training_data_generator.download_current_revision_article_partition_files_in_latest_wikipedia_dump()
# training_data_generator.process_current_revision_article_partition_files()

TRAINING_DATA_FILE_PATH = 'data/training_data.json'

# Builds a generator that yields batches of positive and negative examples each time it is called.
# * Positive examples are randomly sampled true movie and link pairs; they have a label of 1.
# * Negative examples, are randomly sampled false movie and link pairs; they have a label of -1.
def generate_batch(number_of_positive_examples = 50, negative_positive_sample_ratio = 1.0):
  batch_size = number_of_positive_examples * (1 + negative_positive_sample_ratio)
  batch = np.zeros((int(batch_size), 3))

  while True:
    # Randomly choose positive examples.
    for index, (movie_index, link_index) in enumerate(random.sample(movie_link_pairs, number_of_positive_examples)):
      batch[index, :] = (movie_index, link_index, 1)

    index += 1

    # Fill in the rest of the batch with negative examples.
    while index < batch_size:
      random_movie_index = random.randrange(len(training_data))
      random_link_index = random.randrange(len(training_data_all_links))

      # Check to make sure this is not a positive example.
      if (random_movie_index, random_link_index) not in movie_link_pairs:
        batch[index, :] = (random_movie_index, random_link_index, -1)
        index += 1

    np.random.shuffle(batch)

    yield {'movie': batch[:, 0], 'link': batch[:, 1]}, batch[:, 2]

with open(TRAINING_DATA_FILE_PATH, 'r') as training_data_file:
  training_data = json.load(training_data_file)['movies']

# Create a movie to integer index mapping.
training_data_movie_titles_index = {movie['title']: index for index, movie in enumerate(training_data)}

# Get a flat array of all (unique) links and create another index mapping.
training_data_all_links = functools.reduce(operator.concat, [movie['internal_links'] for movie in training_data])
training_data_all_links = list(set(training_data_all_links))
training_data_all_links_index = {link: index for index, link in enumerate(training_data_all_links)}

# Build an array of all movie and link index pairs.
movie_link_pairs = []
for movie in training_data:
  for link in movie['internal_links']:
    movie_link_pairs.append((training_data_movie_titles_index[movie['title']], training_data_all_links_index[link]))

## Define the model
# classification = False
# Both inputs are 1-dimensional
movie = keras.layers.Input(name = 'movie', shape = [1])
link = keras.layers.Input(name = 'link', shape = [1])

# Embedding the movie (shape will be (None, 1, 50))
movie_embedding = keras.layers.Embedding(name = 'movie_embedding', input_dim = len(training_data), output_dim = 50)(movie)
# Embedding the link (shape will be (None, 1, 50))
link_embedding = keras.layers.Embedding(name = 'link_embedding', input_dim = len(training_data_all_links), output_dim = 50)(link)

# Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
merged = keras.layers.Dot(name = 'dot_product', normalize = True, axes = 2)([movie_embedding, link_embedding])

# Reshape to be a single number (shape will be (None, 1))
merged = keras.layers.Reshape(target_shape = [1])(merged)

# Squash outputs for classification
out = keras.layers.Dense(1, activation = 'sigmoid')(merged)
model = Model(inputs = [movie, link], outputs = out)

# Compile using specified optimizer and loss 
model.compile(
  optimizer = 'Adam',
  loss = 'binary_crossentropy',
  metrics = ['accuracy']
)

model.summary()
