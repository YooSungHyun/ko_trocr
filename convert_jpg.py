import glob

from utils.dataset_utils import vector_to_save_image

train_json = glob.glob("data/raw/train/*.json")
train_json.sort()
test_json = glob.glob("data/raw/test/*.json")
test_json.sort()
train = vector_to_save_image(train_json, "data/train_png")
test = vector_to_save_image(test_json, "data/test_png")
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
