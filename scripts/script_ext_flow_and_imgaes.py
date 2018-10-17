import os




def makedirs_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_specific_file_list_from_fd(dir_name, fileType, nameOnly=False):
    list_name = []
    for fileTmp in os.listdir(dir_name):
        file_path = os.path.join(dir_name, fileTmp)
        if os.path.isdir(file_path):
            continue
        elif os.path.splitext(fileTmp)[1] == fileType:
            if nameOnly:
                list_name.append(os.path.splitext(fileTmp)[0])
            else:
                list_name.append(file_path)
    return list_name

def run_main():
    # param
    fps =24
    video_fd_path = '/data1/zfchen/data/ILSVRC/Data/VID/val/'
    output_path = '../output/val'
    makedirs_if_missing(output_path)

    file_name_list = get_specific_file_list_from_fd(video_fd_path, 'mp4')
    for i, file_name in enumerate(file_name_list):
        print('%d /%d\n' %(i, len(file_name_list)))
        video_name = os.path.basename(file_name).split('.')[0]
        mp4_to_jpg(video_name, output_path, fps, filenames)

if __name__=='__main__':
    run_main()
