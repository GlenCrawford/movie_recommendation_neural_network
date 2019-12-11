import os
import requests
import re
import subprocess
import xml.sax
import json
import functools
import operator
from multiprocessing import Pool
from bs4 import BeautifulSoup
import tensorflow as tf
import mwparserfromhell

WIKIPEDIA_DUMPS_BASE_URL = 'https://dumps.wikimedia.org/enwiki/'
WIKIPEDIA_LATEST_DUMP_URL = WIKIPEDIA_DUMPS_BASE_URL + 'latest/'
WIKIPEDIA_ARTICLES_CURRENT_REVISIONS_PARTITION_FILES_REGEX = re.compile(r'^enwiki-latest-pages-articles[\d]+.xml-p[\d]+p[\d]+.bz2$')
WIKIPEDIA_ARTICLES_CURRENT_REVISIONS_PARTITION_FILES_DIRECTORY = '/Users/glen/Projects/machine_learning/movie_recommendation_neural_network/data/wikipedia_dump_articles_current_revision_partition_files/'

class WikipediaXmlHandler(xml.sax.handler.ContentHandler):
  def __init__(self):
    xml.sax.handler.ContentHandler.__init__(self)
    self._buffer = None
    self._values = {}
    self._current_tag = None
    self._article_count = 0
    self._movies = []

  # Characters between the open and close tags.
  def characters(self, content):
    if self._current_tag:
      self._buffer.append(content)

  # Open tag of the element.
  # <title> and <text> are the only two that we care about.
  def startElement(self, name, attrs):
    if name in ('title', 'text'):
      self._current_tag = name
      self._buffer = []

  # Close tag of the element.
  def endElement(self, name):
    if name == self._current_tag:
      self._values[name] = ' '.join(self._buffer)

    if name == 'page':
      self._article_count += 1
      movie = process_wikipedia_article(**self._values)
      # Article is about a movie!
      if movie:
        self._movies.append(movie)

def find_wikipedia_dump_links():
  wikipedia_dumps_index_page = requests.get(WIKIPEDIA_DUMPS_BASE_URL).text
  wikipedia_dumps_index_page_soup = BeautifulSoup(wikipedia_dumps_index_page, 'html.parser')

  # Find all links on the page.
  wikipedia_dumps = [a['href'] for a in wikipedia_dumps_index_page_soup.find_all('a') if a.has_attr('href')]

  return wikipedia_dumps

def find_current_revision_article_partition_file_links_in_latest_wikipedia_dump():
  wikipedia_dump_page = requests.get(WIKIPEDIA_LATEST_DUMP_URL).text
  wikipedia_dump_page_soup = BeautifulSoup(wikipedia_dump_page, 'html.parser')

  # Find all links on the page.
  wikipedia_dump_files = [a['href'] for a in wikipedia_dump_page_soup.find_all('a') if a.has_attr('href')]

  # Filter the links to just the files we want.
  wikipedia_dump_current_article_files = list(filter(WIKIPEDIA_ARTICLES_CURRENT_REVISIONS_PARTITION_FILES_REGEX.match, wikipedia_dump_files))

  return wikipedia_dump_current_article_files

# Warning: This will download many GB of files!
def download_current_revision_article_partition_files_in_latest_wikipedia_dump():
  # Temporay hack on the end of the next line to only get the first file.
  for file in find_current_revision_article_partition_file_links_in_latest_wikipedia_dump()[0:1]:
    tf.keras.utils.get_file(
      (WIKIPEDIA_ARTICLES_CURRENT_REVISIONS_PARTITION_FILES_DIRECTORY + file),
      (WIKIPEDIA_LATEST_DUMP_URL + file)
    )

# Process already-imported files from Wikipedia, each process importing one partition file.
def process_current_revision_article_partition_files():
  file_paths = os.listdir(WIKIPEDIA_ARTICLES_CURRENT_REVISIONS_PARTITION_FILES_DIRECTORY)
  file_paths = list(filter(WIKIPEDIA_ARTICLES_CURRENT_REVISIONS_PARTITION_FILES_REGEX.match, file_paths))

  # Create a pool of workers to execute processes.
  pool = Pool(processes = 1)

  processes_movies = pool.map(process_current_revision_article_partition_file, file_paths)

  pool.close()
  pool.join()

  movies = functools.reduce(operator.concat, processes_movies)
  print(str(movies))

# Process a specific partition file.
def process_current_revision_article_partition_file(file_path):
  wikipedia_xml_handler = WikipediaXmlHandler()
  parser = xml.sax.make_parser()
  parser.setContentHandler(wikipedia_xml_handler)

  for line in subprocess.Popen(['bzcat'], stdin = open((WIKIPEDIA_ARTICLES_CURRENT_REVISIONS_PARTITION_FILES_DIRECTORY + file_path)), stdout = subprocess.PIPE).stdout:
    try:
      parser.feed(line)
    except StopIteration:
      break

    # Temporary hack so we don't waste time while developing.
    if wikipedia_xml_handler._article_count >= 10000:
      break

  print(f'\nFinished processing file: {file_path}')
  print(f'- Searched through {wikipedia_xml_handler._article_count} articles.')
  print(f'- Found {len(wikipedia_xml_handler._movies)} movie(s).')

  return wikipedia_xml_handler._movies

def process_wikipedia_article(title, text, template = 'Infobox film'):
  wikipedia_article_parser = mwparserfromhell.parse(text)

  # Determine whether the article is about a movie or not based on whether it includes the film infobox.
  movie_infobox_matches = wikipedia_article_parser.filter_templates(matches = template)

  if len(movie_infobox_matches) >= 1:
    # Extract information from infobox.
    properties = {param.name.strip_code().strip(): param.value.strip_code().strip() for param in movie_infobox_matches[0].params if param.value.strip_code().strip()}

    # Extract internal wikilinks.
    internal_links = [link.title.strip_code().strip() for link in wikipedia_article_parser.filter_wikilinks()]

    # Extract external links.
    external_links = [link.url.strip_code().strip() for link in wikipedia_article_parser.filter_external_links()]

    return (title, properties, internal_links, external_links)
