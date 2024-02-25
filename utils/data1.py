import tensorflow as tf
import random
import pathlib

#将所有数据存放在同一目录下，
#然后将不同类别的图片分别地存放在各自的类别子目录下
image_dir = r'./shoulder/train'
BATCH_SIZE = 32

def get_image_paths(image_dir:str):
    '''
    获取所有图片路径,例如
    ['flower_photos\\sunflowers\\4895721242_89014e723c_n.jpg', 
    'flower_photos\\dandelion\\4164845062_1fd9b3f3b4.jpg', 
    'flower_photos\\dandelion\\3476759348_a0d34a4b59_n.jpg', 
    'flower_photos\\dandelion\\6972675188_37f1f1d6f6.jpg', 
    'flower_photos\\sunflowers\\8928658373_fdca5ff1b8.jpg']
    '''
    data_path = pathlib.Path(image_dir)
    all_image_paths = list(data_path.glob('*/*'))
    all_image_paths = [str(p) for p in all_image_paths]
    random.shuffle(all_image_paths)
    return all_image_paths


image_paths = get_image_paths(image_dir)    
image_count = len(image_paths)
print(image_count)
print(image_paths[:8])


def get_label_and_index(image_dir:str):
    '''
    获取类别名称及其数字表示，例如
    ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    '''
    data_path = pathlib.Path(image_dir)
    label_names = sorted(item.name for item in data_path.glob('*/') if item.is_dir())
    
    label_index = dict((name,index) for index,name in enumerate(label_names))
    return label_names,label_index


label_names,label_index = get_label_and_index(image_dir)
print(label_names)
print(label_index)

#每个图片路径名称及其数字标签
image_labels = [label_index[pathlib.Path(path).parent.name] for path in image_paths]
for image, label in zip(image_paths[:5], image_labels[:5]):
    print(image, ' --->  ', label)

#创建图片路径及其数字标签的dataset
paths_labels_ds = tf.data.Dataset.from_tensor_slices((image_paths,image_labels))

def load_and_process_from_path_label(image_path,image_label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[256,256])  # 原始图片大小重设为(256, 256)
    image = image/255.0    # 归一化到[0,1]范围
    return image,image_label

    
image_label_ds = paths_labels_ds.map(load_and_process_from_path_label)
print('\n image_label_ds \n',image_label_ds)

def shuffle_image_label_ds(image_label_ds):
    image_label_ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))

    return image_label_ds.batch(BATCH_SIZE)
    

image_label_shuffle_ds = shuffle_image_label_ds(image_label_ds)
print('\n image_label_shuffle_ds \n',image_label_shuffle_ds)