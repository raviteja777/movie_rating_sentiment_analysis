# helper modules
import os
import re
import logging


class FileUtils:
    dirs = []
    files = []

    def update_files_list(self, file_path):
        for f in os.scandir(file_path):
            if f.is_dir():
                self.dirs.append(f.path)
            else:
                pat = re.compile(r"\d+_\d+.txt")
                mat = pat.search(f.path)
                if mat is not None:
                    self.files.append(f.path)
        while len(self.dirs) > 0:
            self.update_files_list(self.dirs.pop())

    def get_files(self):
        return self.files


class LoggerUtil:

    def __init__(self,session,log_file_path):
        self.log_file = os.path.join(log_file_path,session+'.log')
        self.logger = logging
        self.logger.basicConfig(filename=self.log_file,datefmt='%y-%m-%d',level='INFO')

    def log_message(self,message):
        self.logger.log(level=logging.INFO, msg=message)

# get_files_list("data/train")
# print(files)
