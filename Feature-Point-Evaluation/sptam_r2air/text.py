# # import os
# #
# # directory_path = "/home/xcy/euroc/MH_05_difficult/mav0/cam0/data"
# # output_path = "/home/xcy/Desktop/evo/05.txt"
# #
# # # 使用列表解析提取文件名（不带扩展名）
# # file_names = sorted([os.path.splitext(filename)[0] for filename in os.listdir(directory_path) if filename.endswith(".png")],key=int)
# #
# # # 将文件名保存到指定的输出文件中
# # with open(output_path, "w") as f:
# #     for name in file_names:
# #         f.write(name + "\n")
#
# # 定义文件路径
# path_v202 = "/home/xcy/Desktop/evo/10.txt"
# path_tum = "/home/xcy/Desktop/evo/r2d2-euroc/10r.tum"
#
# # 读取v202.txt的内容
# with open(path_v202, 'r') as f:
#     timestamps = f.readlines()
#
# # 对每个时间戳进行转换
# converted_timestamps = ["{:.18e}".format(float(ts.strip()) / 1e9) for ts in timestamps]
#
# # 读取v202.tum的内容
# with open(path_tum, 'r') as f:
#     tum_data = f.readlines()
#
# # 检查两个文件的行数是否匹配
# if len(converted_timestamps) != len(tum_data) :
#     raise ValueError("The number of lines in the two files do not match!")
#
# # 合并转换后的时间戳和TUM数据
# combined_data = []
# for ts, tum in zip(converted_timestamps, tum_data):
#     combined_data.append(ts + " " + tum)
#
# # 保存合并后的数据到v202.tum文件
# with open(path_tum, 'w') as f:
#     for line in combined_data:
#         f.write(line)
def check_tum_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for idx, line in enumerate(lines):
            entries = line.strip().split()

            # Check if there are 8 entries in the line
            if len(entries) != 8:
                print(f"Error on line {idx + 1}: Not 8 entries. Found {len(entries)} entries.")
                return

            # Check for trailing spaces
            if line != line.rstrip():
                print(f"Error on line {idx + 1}: Trailing space or delimiter found.")
                return

        print("No issues found in the file!")


# Specify the path to your TUM file
tum_file_path = "/home/xcy/Desktop/evo/09a.tum"
check_tum_file(tum_file_path)