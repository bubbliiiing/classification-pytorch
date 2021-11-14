import os
from os import getcwd

#---------------------------------------------------#
#   训练自己的数据集的时候一定要注意修改classes
#   修改成自己数据集所区分的种类
#   
#   种类顺序需要和训练时用到的model_data下的txt一样
#---------------------------------------------------#
classes = ["cat", "dog"]
sets    = ["train", "test"]

if __name__ == "__main__":
    wd = getcwd()

    for se in sets:
        list_file = open('cls_' + se + '.txt', 'w')

        datasets_path = "datasets/" + se
        types_name = os.listdir(datasets_path)
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id = classes.index(type_name)
            
            photos_path = os.path.join(datasets_path, type_name)
            photos_name = os.listdir(photos_path)
            for photo_name in photos_name:
                _, postfix = os.path.splitext(photo_name)
                if postfix not in ['.jpg', '.png', '.jpeg']:
                    continue
                list_file.write(str(cls_id) + ";" + '%s/%s'%(wd, os.path.join(photos_path, photo_name)))
                list_file.write('\n')
        list_file.close()

