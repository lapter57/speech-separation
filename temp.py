import sys
sys.path.append("../lib")
import utils
import numpy as np

MIX_TRAIN = "../data/audio_train/mix"
MIX_TEST = "../data/audio_test/mix"
CRM_TRAIN = "../data/audio_train/crm"
CRM_TEST = "../data/audio_test/crm"

mix_train_files = utils.get_files(MIX_TRAIN)
crm_train_files = utils.get_files(CRM_TRAIN)

mix_test_files = utils.get_files(MIX_TEST)
crm_test_files = utils.get_files(CRM_TEST)

for file in mix_test_files:
    a = np.load(file)
    if np.isnan(a).any():
        print("* NAN = " + str(file))
    if np.isinf(a).any():
        print("* INF = " + str(file))

for file in crm_test_files:
    a = np.load(file)
    if np.isnan(a).any():
        print("* NAN = " + str(file))
    if np.isinf(a).any():
        print("* INF = " + str(file))



