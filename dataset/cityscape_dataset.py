from torch.utils.data import Dataset, DataLoader
import os
import glob
import torch as T
from PIL import Image
import json
import numpy as np
import h5py
import io

'''
Macro var
'''
TEMP_MIN = 13.5
TEMP_MAX = 44.5
GPS_LAT_MIN = -90
GPS_LAT_MAX = 90
GPS_LONG_MIN = 0
GPS_LONG_MAX = 180
GPS_HEAD_MIN = 0
GPS_HEAD_MAX = 359
TIMESTAMP_MIN = 1.117480384 # 1117480384
TIMESTAMP_MAX = 6000#6264.143921479001 # original 6264143921479


TARGET_MIN = -1
TARGET_MAX = 1


TRAIN = 'train/'
TEST = 'test/'
VAL = 'val/'


CITIES = ['zurich', # temp boostrap solution for 'h5_train' mode since our root_dir rn will be to .h5 file
 'strasbourg',
 'weimar',
 'aachen',
 'tubingen',
 'jena',
 'bochum',
 'darmstadt',
 'dusseldorf',
 'hamburg',
 'cologne',
 'monchengladbach',
 'krefeld',
 'ulm',
 'hanover',
 'stuttgart',
 'erfurt',
 'bremen']




def get_city(root_dir, mode = TRAIN  ):
  '''
  return a list of str, each one name of city
  '''
  dir = root_dir + 'leftImg8bit_trainvaltest/leftImg8bit/' + mode
  return [name for name in os.listdir(dir) if name != '.DS_Store']


def scale_to_range(x, r_min, r_max, t_min, t_max):
  '''
  scale x to range [t_min, t_max] from [r_min, r_max]

  also make sure number is not out of range

  https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
  
  '''
  scaled = (x-r_min)/(r_max-r_min) *(t_max-t_min) + t_min
  if scaled > t_max:
    scaled = t_max
  if scaled < t_min:
    scaled = -t_min

  return np.float32(scaled)


class Cityscape_Dataset(Dataset):
  '''
  instantiate one of these for train, test and val separately 

  map each 
  
  '''
  def __init__(self, root_dir,x_dir_name ,y_dir_name,meta_data ,mode, input_type, output_type ,transform = None, mode_data = 'normal', device = 'cpu'):
    """
    Assume repo structure

    root_dir
        - /leftImg8bit_trainvaltest: Groud Truth images
        - /vehicle_trainvaltest
        - /gtFine_trainvaltest
        - /camera_trainvaltest
        - /timestamp_sequence



    @parms:
        root_dir: root directoary. There are two possibilities and depeond on "mode_data" options
            1) if 'normal' or 'h5_create_data': root_dir pts to where our actual data sits
            2) if 'h5_train': points to .h5 file 
        meta_data: a [] of str, each entry is the dir name of the meta data to use ie 'vehicle_trainvaltest/vehicle/'
        transform: transformed can be used to call functions on images before being passed into the dataset
        x_dir_name: dir name of input's dir, ie 'gtFine_trainvaltest'
        y_dir_name: dir namae of output's dir
        mode: what mode to use, ie TRAIN, TEST and VAL
        input_type: a suffix that states which image to use for input ie: 'gtFine_color.png'
        mode_data: whether we wannt to use  h5py, since it is faster if we load a big file onto colab -> faster
             There are three modes
            1) 'normal': we read files from paths
            2) 'h5_create_data': this object is used to iterate over the files to create h5 dataset
            3) 'h5_train': we read from h5 from instead now 

    """
    self.root_dir = root_dir
    self.meta_data = meta_data
    self.transform = transform
    self.x_dir_name = x_dir_name
    self.y_dir_name = y_dir_name
    self.mode = mode
    self.input_type = input_type
    self.output_type = output_type
    self.transform = transform
    self.mode_data = mode_data


    # other paths
    self.x_dir = self.root_dir + 'gtFine_trainvaltest/gtFine/' + self.mode 
    self.y_dir = self.root_dir + 'leftImg8bit_trainvaltest/leftImg8bit/' + self.mode

    
    
    self.x_dir_num_file = {} # self.x_dir_num_file['zurich'] = 20, means there are 20 files in zurich
    self.y_dir_num_file = {} # same as above but for y
    
    self.x_idx_to_path = {} # map unique idx for this object to path of image across all cities
    self.num_data = 0 # total numb of data pts we hv to train
    self.metadata_path = {}
    self.city_idx = {} # map city to idx

    

    if self.mode_data == 'normal' or self.mode_data == 'h5_create_data':
      self._get_idx_to_path()
      self._get_meta_paths()
      # other meta info
      self.cities = get_city(self.root_dir, self.mode)
    elif self.mode_data == 'h5_train':
      self.data_file = h5py.File(self.root_dir, 'r') # open a hdf5 file

      # @!!!!!!!!!!!!!!! hard code these two for now 
      self.cities = CITIES
      self.num_data = 2975 

    # fill in city to idx
    c_idx = 0
    for c in self.cities:
      self.city_idx[c] = c_idx
      c_idx += 1


  def _get_idx_to_path(self):
    '''
    fill in the dict that maps idx to path to image
    
    @params:
      input_type: in our case can be 'gtFine_color.png', ''... which image we want to use 

    '''
    # each entry in files, list, is a full path to image ie /Users/...sth in btw.../Dataset/gtFine_trainvaltest/gtFine/train/monchengladbach/monchengladbach_000000_015685_gtFine_color.png
    files = glob.glob( self.x_dir + '*/*_' + self.input_type) # ex: self.x_dir + '*/*_gtFine_color.png'
    
    print(f'files:{len(files)};   {self.x_dir + "*/*_" + self.input_type}')
   

    for i in range(len(files)):
      self.x_idx_to_path[i] = files[i]

    self.num_data = len(files)
    return


  def _get_x_name(self, path_full):
    '''
    each entry in self.x_idx_to_path is a full path ie '/Users/...sth in btw.../Dataset/gtFine_trainvaltest/gtFine/train/monchengladbach/monchengladbach_000000_015685_gtFine_color.png'
    
    but we want to use 'monchengladbach_000000_015685' (call this stem) in 'monchengladbach_000000_015685_gtFine_color.png'
    to access other folders such as y and temperature

    for consistency just extrac this stem from x_idx_to_path bc it might be different in other dir
    
    @params:
      path_full: ie /Users/...sth in btw.../Dataset/gtFine_trainvaltest/gtFine/train/monchengladbach/monchengladbach_000000_015685_gtFine_color.png
    
    @returns:
      png_filename_only_stem: ie 'monchengladbach_000000_015685'
      city_name: 
    '''

    png_filename_only = path_full.split('/')[-1] # ie monchengladbach_000000_015685_gtFine_color.png

    png_filename_only_stem = png_filename_only.split('_')[0:3] # ie ['monchengladbach', '000000', '015685']

    city_name = png_filename_only_stem[0]
    png_filename_only_stem[0] += '_'
    png_filename_only_stem[1] += '_'


    # after add now ['monchengladbach_', '000000_', '015685']

    png_filename_only_stem = "".join(png_filename_only_stem)

    return png_filename_only_stem,city_name





  def _get_meta_paths(self):
    '''
    fill up with paths ie root_dir + 'vehicle_trainvaltest/vehicle/' in metadata_path
    
    ie an entry in self.meta_data looks like: 'vehicle_trainvaltest/vehicle/'

    we fill self.metadata_path['vehicle'] = root_dir + 'vehicle_trainvaltest/vehicle/'

    '''
    for folder in self.meta_data:
      self.metadata_path[folder.split('_')[0]] = self.root_dir + folder + self.mode



  def __len__(self):
    return self.num_data


  def _get_metadata(self, path, curr_meta):
    '''
    grabs metadata

    @modifies:
      self.data
    
    '''
    
    #print('       model normal curr_meta', curr_meta)
    #print('pth to data', data)
    if curr_meta == 'vehicle':
      with open(path, 'r') as f:
        data_file = json.load(f)
        self.data['temperature'] = scale_to_range(data_file['outsideTemperature'] , TEMP_MIN, TEMP_MAX, TARGET_MIN, TARGET_MAX )
        self.data['gpsHeading'] = scale_to_range(data_file['gpsHeading'] , GPS_HEAD_MIN, GPS_HEAD_MAX, TARGET_MIN, TARGET_MAX )
        self.data['gpsLatitude'] = scale_to_range(data_file['gpsLatitude'] , GPS_LAT_MIN, GPS_LAT_MAX, TARGET_MIN, TARGET_MAX )
        self.data['gpsLongitude'] = scale_to_range(data_file['gpsLongitude'] , GPS_LONG_MIN , GPS_LONG_MAX, TARGET_MIN, TARGET_MAX )


    elif curr_meta == 'timestamp' :
      
      with open(path) as f:
        lines = f.readlines()[0].strip('\n')
        # convert to seconds 
        self.data['timestamp'] =  scale_to_range(float(lines)/1e9 , TIMESTAMP_MIN, TIMESTAMP_MAX, TARGET_MIN, TARGET_MAX )


       
    '''
    elif curr_meta == 'camera':
      continue

    else:

      raise Exception
    '''
  

    return 

  def __getitem__(self, idx):
    '''
    @ returns:
      data: for img_x and img_y we return path for  'h5_create_data' mode
    '''
   

    self.data = {}  

    

    if self.mode_data == 'normal' or self.mode_data == 'h5_create_data':
      path_to_data = self.x_idx_to_path[idx]
      data_name_stem, city = self._get_x_name(path_to_data)
      path_to_label = self.y_dir +city + '/'+  data_name_stem + '_' + self.output_type
      self.data['city'] = T.nn.functional.one_hot(T.tensor(self.city_idx[city]), len(self.cities))

      for meta_data in self.metadata_path:
        path_to_meta_data_not_full = self.metadata_path[meta_data] + city + '/' + data_name_stem + '_*'
        path_to_meta_datafull = glob.glob( path_to_meta_data_not_full) [0]
        self._get_metadata(path_to_meta_datafull, meta_data)   

    

    # load images depending on datamode
    if self.mode_data == 'normal':

  
      img_x = Image.open(path_to_data).convert('RGB')
      img_y = Image.open(path_to_label).convert('RGB')

    elif self.mode_data == 'h5_create_data' :
      
      img_x = path_to_data
      img_y = path_to_label


    elif  self.mode_data == 'h5_train':
      img_x_binary = np.array(self.data_file['X'][idx])  
      img_y_binary = np.array(self.data_file['Y'][idx])  

      img_x = Image.open(io.BytesIO(img_x_binary)).convert('RGB')
      img_y = Image.open(io.BytesIO(img_y_binary)).convert('RGB')

      self.data['city'] = self.data_file['city'][idx]
      self.data['temperature'] = self.data_file['temperature'][idx][0]
      self.data['gpsHeading'] = self.data_file['gpsHeading'][idx][0]
      self.data['gpsLatitude'] = self.data_file['gpsLatitude'][idx][0]
      self.data['gpsLongitude'] = self.data_file['gpsLongitude'][idx][0]
      self.data['timestamp'] = self.data_file['timestamp'][idx][0]






    if self.transform:
      img_x = self.transform(img_x)
      img_y = self.transform(img_y)

    self.data['A'] = img_x
    self.data['B'] = img_y
    
    #return self.data  

    # we only need A and B for cycleGAN
    return self.data['A'].to(device), self.data['B'].to(device)