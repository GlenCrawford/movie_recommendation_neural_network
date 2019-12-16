import json
import functools
import operator
import training_data_generator

# training_data_generator.download_current_revision_article_partition_files_in_latest_wikipedia_dump()
# training_data_generator.process_current_revision_article_partition_files()

TRAINING_DATA_FILE_PATH = 'data/training_data.json'

with open(TRAINING_DATA_FILE_PATH, 'r') as training_data_file:
  training_data = json.load(training_data_file)['movies']

# creata a mapping from movie to int
training_data_movie_titles_index = {movie['title']: index for index, movie in enumerate(training_data)}

# get a flat array of all links and create index
training_data_all_links = functools.reduce(operator.concat, [movie['internal_links'] for movie in training_data])
training_data_all_links_index = {link: index for index, link in enumerate(training_data_all_links)}


