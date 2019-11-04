from os.path import dirname, realpath
from os import listdir
from os.path import isfile, join
from shutil import copyfile

path = '/home/simon/datasets/DirtDataset/ipa_data/InputAndGroundTruth/'
list = ['carpet-clean',
        'carpet-food',
        'carpet-fuzz',
        'carpet-leaves',
        'carpet-office',
        'carpet-paper1',
        'carpet-paper2',
        'carpet-stones',
        'carpet-streetdirt',
        'corridor-clean',
        'corridor-paper2',
        'corridor-stones',
        'kitchen-clean',
        'kitchen-fuzz',
        'kitchen-paper2',
        'linoleum-clean',
        'linoleum-food',
        'linoleum-fuzz',
        'linoleum-leaves',
        'linoleum-office',
        'linoleum-paper1',
        'linoleum-paper2',
        'linoleum-paper3',
        'office-clean',
        'office-food',
        'office-leaves',
        'office-office',
        'office-paper1',
        'office-paper2',
        'office-paper3',
        'office-stones',
        'office-streetdirt']

path_target = 'everything_combined'

gt_tgt = path + path_target + '/groundtruth/'
im_tgt = path + path_target + '/images/'
count = 0
for folder in list:
    gt_dir = path + folder + '/GroundTruth/'
    im_dir = path + folder + '/Original/'
    onlyfiles = [f for f in listdir(gt_dir) if isfile(join(gt_dir, f))]
    for file in onlyfiles:
        copyfile(gt_dir+file, gt_tgt + str(count) + '.png')
        copyfile(im_dir+file, im_tgt + str(count) + '.png')
        count = count + 1

