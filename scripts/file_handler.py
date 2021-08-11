import os
import random
from app import load_logging


class FileHandler():
  """Read audio, audio transcription, Save cleaned Audio and transcriptions
  """
  def __init__(self):
    self.logger = load_logging("FileHandler")


  def read_data(self, PATH_TEST_WAV):
      label_list = []
      try:
        test_labels = os.listdir(PATH_TEST_WAV)
        test_labels = [i.strip('.wav') for i in test_labels]
        for i in range(20):
            n = random.randint(1,350)
            label_list.append(test_labels[n])

        return label_list

      except Exception as e:
        #   pass
        self.logger.exception(f" Exception occured in loading sample audio file, {e}")

