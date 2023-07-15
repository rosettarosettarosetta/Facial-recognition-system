import os
from shutil import copy2
import random


# 通过这个函数来划分数据集到user_data文件夹中
def dataset_split(src_data_folder, target_data_folder, train_scale=0.8, test_scale=0.2):
    '''
    生成划分好的训练集、验证集和测试集
    :param test_scale: 测试集比例
    :param src_data_folder: 源文件夹（colling_dataset）
    :param target_data_folder: 目标文件夹（user_data）
    :param train_scale: 训练集比例
    :return: 
    '''

    # 开始数据集划分
    print("开始数据集划分")

    if not os.path.exists("./user_data"):
        os.mkdir("./user_data")

    class_names = os.listdir(src_data_folder)
    # 在目标文件夹中创建训练集、测试集文件夹
    split_names = ["train", "test"]
    for split_name in split_names:
        split_path = os.path.join(target_data_folder, split_name)
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        # 然后在split——path中创建各个类别文件夹
        for class_name in class_names:
            class_split_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_split_path):
                pass
            else:
                os.mkdir(class_split_path)

    # 按照比例划分数据集，并进行复制
    # 首先进行分类遍历
    for class_name in class_names:
        current_class_data_path = os.path.join(src_data_folder, class_name)
        # 获取当前类别下的所有图片
        current_all_data = os.listdir(current_class_data_path)
        current_data_len = len(current_all_data)
        current_data_index_list = list(range(current_data_len))
        random.shuffle(current_data_index_list)

        train_folder = os.path.join(os.path.join(target_data_folder, "train"), class_name)
        test_folder = os.path.join(os.path.join(target_data_folder, "../test"), class_name)
        train_stop_flag = current_data_len * train_scale
        test_stop_flag = current_data_len * (train_scale + test_scale)

        current_idx = 0
        train_num = 0
        test_num = 0

        for i in current_data_index_list:
            src_img_path = os.path.join(current_class_data_path, current_all_data[i])
            if current_idx <= train_stop_flag:
                copy2(src_img_path, train_folder)
                train_num += 1
            else:
                copy2(src_img_path, test_folder)
                test_num += 1

            current_idx += 1

        print(f"类别{class_name}按照{train_scale},{test_scale}划分完成,一共{current_data_len}张图片")
        print(f"训练集{train_folder}:{train_num}张")
        print(f"测试集{test_folder}:{test_num}张")


if __name__ == "__main__":
    src_data_folder = "./calling_dataset"
    target_data_folder = "./user_data"

    # 调用dataset_split函数进行数据集划分
    # 可以按照需要调整训练集和测试集的比例
    dataset_split(src_data_folder, target_data_folder, train_scale=0.8, test_scale=0.2)

    print("数据集划分完成！")

