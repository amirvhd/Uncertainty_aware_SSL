from __future__ import print_function
from torchvision.datasets.utils import download_and_extract_archive


def download():

    url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip'
    download_and_extract_archive(url, download_root="./datasets/ISIC_data")
    print("download is finished")
    url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip'
    download_and_extract_archive(url, download_root="./datasets/ISIC_data")
    print("download is finished")
    url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_Input.zip'
    download_and_extract_archive(url, download_root="./datasets/ISIC_data")
    print("download is finished")
    url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Validation_GroundTruth.zip'
    download_and_extract_archive(url, download_root="./datasets/ISIC_data")
    print("download is finished")
    url = 'https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Test_Input.zip'
    download_and_extract_archive(url, download_root="./datasets/ISIC_data")
    print("download is finished")

def set_path():
    train_img_path = './datasets/ISIC_data/ISIC2018_Task3_Training_Input'
    train_class_path = './datasets/ISIC_data/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'

    val_img_path = './datasets/ISIC_data/ISIC2018_Task3_Validation_Input'
    val_class_path = './datasets/ISIC_data/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv'

    test_img = './datasets/ISIC_data/ISIC2018_Task3_Test_Input'
    return train_img_path, train_class_path, val_img_path, val_class_path, test_img
    # classes = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']


def main():
    download()


if __name__ == '__main__':
    main()
