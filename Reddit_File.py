import os
import shutil

# 设置原始文件夹和目标文件夹路径
source_folder = '/Users/samxie/Research/HEC/Reddit Sentiment 0417/Data_Sen'
destination_folder = '/Users/samxie/Research/HEC/Reddit Sentiment 0417/Data_Done'

# 遍历源文件夹中的所有文件
for file_name in os.listdir(source_folder):
    # 检查文件是否为CSV文件
    if file_name.endswith('.csv'):
        # 获取文件前缀（不包括后缀）
        prefix = file_name.split('_')[0]
        
        # 创建目标文件夹（如果不存在的话）
        target_folder = os.path.join(destination_folder, prefix)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # 移动文件到目标文件夹
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(target_folder, file_name)
        shutil.move(source_file, destination_file)

print("文件已成功移动到对应的文件夹")