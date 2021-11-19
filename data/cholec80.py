import os
import io
import numpy as np
from PIL import Image
from imageio import imread


class Cholec80(object):
    def __init__(self, data_root, train=True, seq_len=20, image_size=64):
        self.root_dir = data_root
        if train:
            self.video_ids = list(range(1, 41))
            self.ordered = False
        else:
            self.video_ids = list(range(41, 81))
            self.ordered = True
        self.dirs = []
        for i in self.video_ids:
            curr_seq_len = 0
            curr_dir = []
            data_dir = f'{self.root_dir}/frame/{i}'
            for d in sorted(os.listdir(data_dir)):
                curr_dir.append('%s/%s' % (data_dir, d))
                curr_seq_len += 1
            self.dirs.append(curr_dir)
        self.seq_len = seq_len
        self.image_size = image_size
        self.seed_is_set = False  # multi threaded loading
        self.video_id = self.video_ids[0]
        self.frame_id = 0

    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)

    def __len__(self):
        return len(self.dirs)

    def get_seq(self):
        if self.ordered:
            if self.frame_id == len(self.dirs[self.video_id]) - 1:
                self.frame_id = 0
                self.video_id = (self.video_id + 1) % len(self.dirs)
            else:
                self.frame_id += 1
        else:
            self.video_id = np.random.randint(len(self.dirs))
            self.frame_id = np.random.randint(len(self.dirs[self.video_id]) - self.seq_len)
        image_seq = []
        for i in range(self.seq_len):
            fname = self.dirs[self.video_id][self.frame_id]
            im = imread(fname).reshape(1, 64, 64, 3)
            image_seq.append(im / 255.)
        image_seq = np.concatenate(image_seq, axis=0)
        return image_seq

    def __getitem__(self, index):
        self.set_seed(index)
        return self.get_seq()


