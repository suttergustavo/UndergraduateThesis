import os
import glob
import keras
import numpy as np
import pandas as pd
from PIL import Image

class UnbalancedDataLoader(keras.utils.Sequence):
    def __init__(self, data_root, classes_dist, target_size=(299, 299), n_channels=3, batch_size=32, 
                 normalize=([0, 0, 0], [1, 1, 1]), data_aug=False, noise=0, repeat=False, shuffle=True,
                 random_state=None):
        
        self.target_size = target_size
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.shuffle = shuffle
        self.normalize = {'mean': normalize[0], 'scale': normalize[1]}
        self.noise = noise
        
        self.path_list = prepare_path_list(data_root, classes_dist, repeat, random_state, pd_return=False)
        
        self.indexes = np.arange(self.path_list.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        return int(np.ceil(self.indexes.shape[0] / self.batch_size))
    
    def __getitem__(self, batch_index):
        idx_start = batch_index * self.batch_size
        idx_end = (batch_index + 1) * self.batch_size
        batch_indexes = self.indexes[idx_start: idx_end]
        
        batch_files = self.path_list[batch_indexes]
            
        batch = np.empty((self.batch_size, *self.target_size, self.n_channels))
                
        for i, file in enumerate(batch_files):
            img = Image.open(file)
            if img.mode != 'RGB' and self.n_channels == 3:
                img = img.convert('RGB')

            img = img.resize(self.target_size)
            if self.data_aug != False:
                if np.random.uniform() < self.data_aug['flip_prob']:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
                rotate_angle = np.random.uniform(0, self.data_aug['max_angle'])
                img = img.rotate(rotate_angle)       
            
            img = np.array(img)
            img = img + self.noise * np.random.normal(0, 255, size=img.shape)
            img = img / 255
            img = np.clip(img, 0, 1)
                       

            img = (img - self.normalize['mean'])/self.normalize['scale']
            
            batch[i, :, :, :] = img
            
        return batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
        
def prepare_path_list(file_path, classes_dict, repeat=False, seed=None, pd_return=True):
    if seed != None: np.random.seed(seed)
    
    selected_paths = np.array([])
    targets = np.array([])
    
    for cls, n in classes_dict.items():
        path_class_files = os.path.join(file_path, cls, "*")
        path_class_files = np.array(glob.glob(path_class_files))
        
        if n == 'all': 
            n = path_class_files.shape[0]
        
        selected_paths_class = np.random.choice(path_class_files, size=n, replace=repeat)

        selected_paths = np.concatenate((selected_paths, selected_paths_class))
        targets = np.concatenate((targets, n * [cls]))

    
    if pd_return:
        return pd.DataFrame(data=np.vstack((selected_paths, targets)).T, 
                            columns=['filename', 'class'])
    return selected_paths


def prepare_path_list_celeba(data_root, csv_filename, attr_list, repeat=False, seed=None):
    if seed != None: np.random.seed(seed)
    df = pd.read_csv(csv_filename)
    
    selected_paths = np.array([])
    targets = np.array([])
    
    for attr, n in attr_list:
        query = ' and '.join([f'{a} == {v}' for a, v in attr.items()])
        df_attr = df.query(query)
        path_class_files = df_attr['image_id']
        
        if n == 'all':
            n = path_class_files.shape[0]
        
        random_idx = np.random.choice(df_attr.index, size=n, replace=repeat)
        
        selected_paths_class = path_class_files[random_idx]
        selected_paths_class = selected_paths_class.apply(lambda x: os.path.join(data_root, x))
        selected_paths_class = selected_paths_class.values
        
        selected_paths = np.concatenate([selected_paths, selected_paths_class])
        targets = np.concatenate([targets, n * [query]])
        
    return pd.DataFrame(data=np.vstack((selected_paths, targets)).T, 
                        columns=['filename', 'class'])


def get_celeba_attributes():
    return ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
            'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 
            'Black_Hair','Blond_Hair', 'Blurry', 'Brown_Hair', 
            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
            'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
            'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Young']


def examples_per_class(root_folder):
    classes_paths = glob.glob(os.path.join(root_folder,'*'))
    classes_names = [os.path.split(c)[-1] for c in classes_paths]
    classes_n = [len(glob.glob(os.path.join(c, '*'))) for c in classes_paths]
    
    return dict(zip(classes_names, classes_n))