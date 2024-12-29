import os
from tqdm import tqdm


def construct_frame_groups(base_path):
    # 定义frame和groundtruth的路径
    frame_path = os.path.join(base_path, 'frame')
    groundtruth_path = os.path.join(base_path, 'groundtruth')

    # 使用base_path的最后一部分来命名输出的txt文件
    base_name = os.path.basename(os.path.normpath(base_path))
    output_file_path = os.path.join(base_path, f'{base_name}.txt')

    total_frame_groups = 0  # 用于统计五帧组的总数

    # 打开文件以写入
    with open(output_file_path, 'w') as output_file:
        # 获取所有视频文件夹
        video_folders = [folder for folder in os.listdir(frame_path) if os.path.isdir(os.path.join(frame_path, folder))]

        # 使用tqdm显示进度条
        for video_folder in tqdm(video_folders, desc="Processing videos"):
            video_frame_dir = os.path.join(frame_path, video_folder)
            video_groundtruth_dir = os.path.join(groundtruth_path, video_folder)

            if os.path.isdir(video_frame_dir) and os.path.isdir(video_groundtruth_dir):
                # 获取帧文件列表并排序
                frame_files = sorted(os.listdir(video_frame_dir))

                # 遍历寻找五帧组
                for i in range(len(frame_files) - 4):
                    # 获取五帧的路径
                    frame_group = [os.path.join('frame', video_folder, frame_files[j]) for j in range(i, i + 5)]
                    groundtruth_frame = os.path.join('groundtruth', video_folder,
                                                     frame_files[i + 2])  # 中间帧对应的groundtruth

                    # 生成输出行
                    line = ' '.join(frame_group) + ' ' + groundtruth_frame + '\n'
                    output_file.write(line)

                    total_frame_groups += 1  # 增加计数器

    print(f"Total number of 5-frame groups constructed: {total_frame_groups}")
    print(f"Frame groups have been written to {output_file_path}")


# 调用函数，输入文件夹A的路径
construct_frame_groups(r'H:\SPL\MOSE100\E2FGVI_large')
construct_frame_groups(r'H:\SPL\MOSE100\E2FGVI_medium')
construct_frame_groups(r'H:\SPL\MOSE100\E2FGVI_small')

construct_frame_groups(r'H:\SPL\MOSE100\FuseFormer_large')
construct_frame_groups(r'H:\SPL\MOSE100\FuseFormer_medium')
construct_frame_groups(r'H:\SPL\MOSE100\FuseFormer_small')

construct_frame_groups(r'H:\SPL\MOSE100\STTN_large')
construct_frame_groups(r'H:\SPL\MOSE100\STTN_medium')
construct_frame_groups(r'H:\SPL\MOSE100\STTN_small')