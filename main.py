from sys import argv
import json
import functools
import operator
import datetime
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import training_data_generator

TRAINING_DATA_FILE_PATH = 'data/training_data.json'
LOG_DIRECTORY = 'logs/fit/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

# Not really how dictionaries are meant to be used, but better than building a second (reversed) dictionary for each index.
def reverse_index_lookup(index, value):
  return list(index.keys())[list(index.values()).index(value)]

# Builds a generator that yields batches of positive and negative examples each time it is called.
# * Positive examples are randomly sampled true movie and link pairs; they have a label of 1.
# * Negative examples are randomly sampled false movie and link pairs; they have a label of -1.
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

# Extract and normalize embeddings from the specified layer.
def extract_weights_from_embedding_layer(model, layer_name):
  layer = model.get_layer(layer_name)
  weights = layer.get_weights()[0]
  weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
  return weights

# Receives a movie or link and a set of embeddings, and returns the most similar entities.
# Do this by computing the dot product between the entity queries and the embeddings, which is the the cosine similarity between two vectors.
# With cosine similarity, higher numbers indicate entities that are closer together, with -1 the furthest apart and +1 closest together, so all we have to do to find the closest/furthest entities in the embedding space is sort by the dot products.
def find_similar_entities(name, weights, number_of_results = 10, plot = False):
  try:
    entity_distances = np.dot(weights, weights[training_data_movie_titles_index[name]])
  except KeyError:
    print(f'{name} not found in movie index.')
    return

  sorted_entity_distances = np.argsort(entity_distances)

  # Optionally plot results.
  if plot:
    plot_similar_entities(name, number_of_results, entity_distances, sorted_entity_distances)

  # Find the most similar entities by taking the last n sorted distances.
  closest_entities = sorted_entity_distances[-number_of_results:]

  return entity_distances, closest_entities

def plot_similar_entities(name, number_of_results, entity_distances, sorted_entity_distances):
  # Find furthest and closest entities.
  furthest_entities = sorted_entity_distances[:(number_of_results // 2)]
  closest_entities = sorted_entity_distances[-number_of_results-1: len(entity_distances) - 1]

  entities_to_plot = [reverse_index_lookup(training_data_movie_titles_index, entity) for entity in furthest_entities]
  entities_to_plot.extend(reverse_index_lookup(training_data_movie_titles_index, entity) for entity in closest_entities)

  # Get the distances of the entities to plot.
  distances = [entity_distances[entity] for entity in furthest_entities]
  distances.extend(entity_distances[entity] for entity in closest_entities)

  colors = ['r' for _ in range(number_of_results // 2)]
  colors.extend('g' for _ in range(number_of_results))

  data = pd.DataFrame({'distance': distances}, index = entities_to_plot)

  # Plot a horizontal bar chart.
  data['distance'].plot.barh(color = colors, figsize = (10, 8), edgecolor = 'k', linewidth = 2)
  plt.xlabel('Cosine Similarity')
  plt.axvline(x = 0, color = 'k')

  chart_name = f'Movies Most and Least Similar to'
  for word in name.split():
    chart_name += ' $\it{' + word + '}$'
  plt.title(chart_name, x = 0.2, size = 28, y = 1.05)

  plt.style.use('fivethirtyeight')
  plt.rcParams['font.size'] = 15

  plt.show()

if len(argv) <= 1:
  print('You must pass in a movie name as an argument. Please consult the README.')
  exit()

### Load, index and combine the training data ###

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

### Define the model ###

# Inputs (both one-dimensional).
movie_input = keras.layers.Input(name = 'movie', shape = [1])
link_input = keras.layers.Input(name = 'link', shape = [1])

# Embeddings for the movies (shape will be (None, 1, 50)).
movie_embedding = keras.layers.Embedding(name = 'movie_embedding', input_dim = len(training_data_movie_titles_index), output_dim = 50)(movie_input)

# Embeddings for the links (shape will be (None, 1, 50)).
link_embedding = keras.layers.Embedding(name = 'link_embedding', input_dim = len(training_data_all_links_index), output_dim = 50)(link_input)

# Merge the layers with a dot product along the second axis (shape will be (None, 1, 1)).
# Then reshape to be a single number (shape will be (None, 1)).
merged_output = keras.layers.Dot(name = 'dot_product', normalize = True, axes = 2)([movie_embedding, link_embedding])
merged_output = keras.layers.Reshape(target_shape = [1])(merged_output)

model = keras.Model(inputs = [movie_input, link_input], outputs = merged_output)
model.compile(optimizer = 'Adam', loss = 'mse')

print('\nModel architecture:')

model.summary()

### Train! ###

print('\nTraining the model...')

model.fit_generator(
  generate_batch(number_of_positive_examples = 1024, negative_positive_sample_ratio = 2.0),
  epochs = 15,
  steps_per_epoch = (len(movie_link_pairs) // 1024),
  verbose = 2,
  callbacks = [tf.keras.callbacks.TensorBoard(log_dir = LOG_DIRECTORY)]
)

### Inspect and output the results ###

movie_query = argv[1]

movie_weights = extract_weights_from_embedding_layer(model, 'movie_embedding')
entity_distances, closest_entities = find_similar_entities(movie_query, movie_weights, plot = False)

print(f'\nMovies most similar to {movie_query}:\n')

# Output formatting.
max_width = max([len(reverse_index_lookup(training_data_movie_titles_index, entity)) for entity in closest_entities])

# Print the most similar entitites and their distances.
for entity in reversed(closest_entities):
  print(f'Movie: {reverse_index_lookup(training_data_movie_titles_index, entity):{max_width + 2}} Similarity: {entity_distances[entity]:.{2}}')
