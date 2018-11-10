import subprocess
import os
import os.path as osp
from tqdm import tqdm

target_path = '/hdd/datasets/moments/Moments_in_Time_256x256_30fps/training'


for action_folder in os.listdir(target_path):
    action_example_folder = osp.join(target_path, action_folder)

    for action_example in tqdm(os.listdir(action_example_folder)):
        if not action_example.endswith('.mp4'):
            continue
        action_example_name = action_example.split('.')[0]

        write_dir = osp.join(action_example_folder, action_example_name)
        action_example_path = osp.join(action_example_folder, action_example)

        if not osp.exists(write_dir):
            os.makedirs(write_dir)

        run_cmd = 'ffmpeg -loglevel panic -i %s %s' % (action_example_path, write_dir + '/image%d.png')
        os.system(run_cmd)

