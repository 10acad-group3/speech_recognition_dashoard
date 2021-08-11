import os
from app import load_logging

PATH_TEST_WAV = "../data/AMHARIC/test/wav/"


class FileHandler():
  """Read audio, audio transcription, Save cleaned Audio and transcriptions
  """

  def read_data(PATH_TEST_WAV):
      
      try:
        test_labels = os.listdir(PATH_TEST_WAV)
        test_labels = [i.strip('.wav') for i in test_labels]

        return test_labels

      except Exception as e:
        #   pass
        logging.exception(f" Exception occured in loading sample audio file, {e}")

