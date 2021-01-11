# helper modules
import os
import re


class Utils:
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

# get_files_list("data/train")
# print(files)
