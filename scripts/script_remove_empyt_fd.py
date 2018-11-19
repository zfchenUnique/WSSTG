import os
import pdb

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def full_listdir(dir_name):
    return [os.path.join(dir_name, i) for i in os.listdir(dir_name)]

def get_list_dir(dir_name):
    fileFdList = full_listdir(dir_name)
    folder_list = list()
    for item in fileFdList:
        if os.path.isdir(item):
            folder_list.append(item)
    return folder_list

def clean_tensorboard_folder():
    clean_folder = '../data/tensorBoardX'
    thre = 200
    folder_list = get_list_dir(clean_folder)
    #pdb.set_trace()
    for i, folder_path in enumerate(folder_list):
        folder_size = get_size(folder_path)
        if folder_size<thre:
            print('remove folder %s\n' %(folder_path))
            cmd_line = 'rm %s -r' % (folder_path.replace(' ', '\ '))
            os.system(cmd_line)

if __name__=='__main__':
    clean_tensorboard_folder()
