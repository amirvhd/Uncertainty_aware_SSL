import os
import urllib.request
import zipfile


def download_dataset():
    #if not os.path.isdir("./datasets"):
    #os.makedirs("./datasets")
    print('Beginning dataset download with urllib2')
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    os.chdir("./datasets")
    path = "%s/tiny-imagenet-200.zip" % os.getcwd()
    urllib.request.urlretrieve(url, path)
    print("Dataset downloaded")


def unzip_data():
    os.chdir("./datasets")
    path_to_zip_file = "%s/tiny-imagenet-200.zip" % os.getcwd()
    directory_to_extract_to = os.getcwd()
    print("Extracting zip file: %s" % path_to_zip_file)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print("Extracted at: %s" % directory_to_extract_to)


def split_data():
    os.chdir("./datasets")
    DATA_DIR = 'tiny-imagenet-200'  # Original images come in shapes of [3,64,64]
    # Define training and validation data paths
    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'val')
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    val_img_dir = os.path.join(VALID_DIR, 'images')
    # Open and read val annotations text file
    fp = open(os.path.join(VALID_DIR, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    # Create dictionary to store img filename (word 0) and corresponding
    # label (word 1) for every line in the txt file (as key value pair)
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(val_img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(val_img_dir, img)):
            os.rename(os.path.join(val_img_dir, img), os.path.join(newpath, img))


def main():
    #download_dataset()
    unzip_data()
    split_data()


if __name__ == '__main__':
    main()
