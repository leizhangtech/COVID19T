DECOMPRESS_DIRECTORY= '/media/yanwe/comp/Data/Lung/3D/ICCV/MIA-COV19-DATA'
save_path = '/media/yanwe/新加卷1/ICCV_Lung_split'

from os.path import join,isfile,isdir,dirname,basename,exists
from PIL import Image
import numpy as np
import SimpleITK as sitk
from lungmask import mask
from scipy import ndimage
from math import ceil
import json
import matplotlib.pyplot as plt
import random
import os
from concurrent.futures import ThreadPoolExecutor
import logging
import csv
import gdown
import tarfile
import rarfile
from tqdm import tqdm
import requests

if not exists(save_path):
    os.makedirs(save_path)

blacklist=[r'train/non-covid/ct_scan_853',
           'train/non-covid/ct_scan_537',
           'train/non-covid/ct_scan_354',
           'train/non-covid/ct_scan_292',
           'train/covid/ct_scan_47',
           'train/covid/ct_scan_31',
           'val/non-covid/ct_scan_15',
           'val/covid/ct_scan_48',
           'val/covid/ct_scan_46',
           'val/covid/ct_scan_40',
           'val/covid/ct_scan_18',
           'val/covid/ct_scan_101',
           'val/non-covid/ct_scan_9',
           'test/b98e8b7f-ccef-4d9f-a71f-ff8eab6a3d5a',
           'test/b9a293fa-18e5-4488-ab9d-eab9e7657d23',
           'test/4233de62-d03f-4014-8d1a-0d269cf2cbce', #32.jpg
           'test/11bffa9c-e43a-4521-ba18-5fb5d683b512']



class Dataset_provider(object):
    def __init__(self,data_name,modes = None):
        data_list = {
            'ICCV_Lung': {
                'path': DECOMPRESS_DIRECTORY,
                'type': ['lung', '3D'],
                'label':{'covid':1,'non-covid':0}
            }
        }
        if modes is None:
            modes = ['train', 'val', 'test']
        logging.info('Dataset: '+data_name)
        self.info=data_list[data_name]
        self.dir_path = self.info['path']
        self.type = self.info['type']
        self.split_by_dirs(modes)

    def get_from_json(self, modes, json_file):
        return_tpl= [None, None, None]
        for i in range(0,3):
            if not modes[i]==None:
                return_tpl[i]=json_file[modes[i]]
        return return_tpl


    def split_by_dirs(self, modes):
        json_file = {}
        dir_path={}
        i=0
        for mode in modes:
            if mode:
                if exists(join(self.dir_path, mode)):
                    dir_path[mode] = join(self.dir_path, mode)
                    json_file[mode] = []
                    if mode == 'test':
                        images = [{"image":join(dir_path[mode], f)}  for f in os.listdir(dir_path[mode]) if isdir(join(dir_path[mode], f))]
                        json_file[mode] += images
                    else:
                        classes=[f for f in os.listdir(dir_path[mode]) if isdir(join(dir_path[mode], f))]
                        for cla in classes:
                            images =[{"image":join(dir_path[mode],cla,f),"label": self.info['label'][cla]} for f in os.listdir(join(dir_path[mode],cla)) if isdir(join(join(dir_path[mode],cla), f))]
                            json_file[mode]+=images

                else:
                    logging.warning('Can not find '+mode+' directory. '+join(self.dir_path, mode))
                    modes[i]=None
            i+=1
        with open(join(self.dir_path,'dataset.json'), 'w') as fp:
            json.dump(json_file, fp,indent=2)
        [self.train, self.val, self.test] = self.get_from_json(modes,json_file)




class dataloader(object):
    def __init__(self,file_path_list,dir_path=None,type=None,n_start=0):
        self.dir_path=dir_path
        self.file_path_list=file_path_list
        self.type=type
        self.n_start=n_start
    def __iter__(self):
        self.n = self.n_start
        return self

    def __next__(self):
        if self.n < len(self.file_path_list):
            file_path=self.file_path_list[self.n]
            for black in blacklist:
                if file_path['image'].endswith(black):
                    self.n += 1
                    file_path = self.file_path_list[self.n]
            self.n += 1
            if type(file_path) == dict:
                if 'image' in file_path:
                    image_path=join(self.dir_path,file_path['image'])
                else:
                    raise NotImplemented
                if 'label' in file_path:
                    if type(file_path['label']) is int:
                        label = file_path['label']
                    elif type(file_path['label']) is str:
                        label = join(self.dir_path, file_path['label'])
                    else:
                        raise NotImplemented
                else:
                    label = None
                logging.info(str(self.n) + '/' + str(len(self.file_path_list)) + ' ' + file_path['image'])
                return self.image_loader(image_path, label)

            elif type(file_path) == str:
                image_path = join(self.dir_path, file_path)
                logging.info(str(self.n) + '/' + str(len(self.file_path_list)) + ' ' + file_path)
                return self.image_loader(image_path)
            else:
                raise NotImplemented
        else:
            raise StopIteration

    def image_loader(self, image_path, label=None):
        np_label=np_mask=bbox=sitk_image=sitk_mask=None
        if isfile(image_path):
            if image_path.endswith('.nii.gz'):
                sitk_image = sitk.ReadImage(image_path)
                np_image=sitk.GetArrayFromImage(sitk_image)     # z,y,x
                if 'lung' in self.type:
                    bbox, np_mask = self.get_mask_bbox(sitk_image, objs_amount=2, padding=3, image_path=image_path)
                    sitk_mask=sitk.GetImageFromArray(np_mask)
                    sitk_mask.SetSpacing(sitk_image.GetSpacing())
            elif image_path.endswith('.png'):
                pil_image = Image.open(image_path)
                np_image = np.array(pil_image)
                '''
                PIL: pil_image.size: (x,y) (width,height)
                numpy: (y,x,c) (height,width,channel)
                '''
            else:
                raise NotImplemented

        elif isdir(image_path):
            slices= [f for f in os.listdir(image_path) if isfile(join(image_path, f))]
            zmax= max([int(f.split('.')[0]) for f in slices if not f.startswith('.')])
            images=[]
            for i in range(zmax,-1,-1):
                im_slice = np.array(Image.open(join(image_path, str(i) + '.jpg')))
                res_max=500
                res_min=-1024
                np_slice = (im_slice.astype(np.float16) - 20) / 255 * (res_max - res_min) + res_min
                images.append(np_slice)
            shapes = [image.shape for image in images]
            slices_num = len(shapes)
            most_common_size=max(shapes, key=shapes.count)
            if not most_common_size==(512,512):
                logging.warning(
                    image_path.split(self.dir_path)[1] + ' is not shape (512, 512)' +', its shape is ' + str(most_common_size))
            for shape in shapes[::-1]:
                slices_num=slices_num-1
                if not shape==most_common_size:
                    logging.warning( image_path.split(self.dir_path)[1]+'/'+str(slices_num+1)+'.jpg is not shape '+ str(most_common_size) +', its shape is '+str(shape))
                    images.pop(slices_num)

            np_image=np.stack(images)
            sitk_image = sitk.GetImageFromArray(np_image)
            bbox, np_mask = self.get_mask_bbox(sitk_image, objs_amount = 2, padding = 3, image_path=image_path)
            sitk_image.SetSpacing((0.725,0.725,270/(bbox[0][1]-bbox[0][0])))
            sitk_mask = sitk.GetImageFromArray(np_mask)
            sitk_mask.SetSpacing(sitk_image.GetSpacing())
        else:
            raise NotImplemented


        if not label==None:
            if type(label) is int:
                np_label=label
            elif label.endswith('.nii.gz'):
                sitk_label = sitk.ReadImage(label)
                np_label = sitk.GetArrayFromImage(sitk_label)
            elif label.endswith('.png'):
                pil_label = Image.open(label)
                np_label = np.array(pil_label)
            else:
                raise NotImplemented
            '''
            SimpleITK: sitk_image.GetSize(): (x,y,z) (width,height,depth)
            numpy: np_image.shape: (z,y,x) 
            
            SimpleITK: sitk_image.GetSize(): (x,y) (width,height)
            numpy: np_image.shape: (y,x,c) (height,width,channel)
            '''
        return_dict = {
            'image': np_image,
            'label': np_label,
            'lung_mask': np_mask,
            'sitk_lung_mask': sitk_mask,
            'bbox': bbox,
            'sitk_im':sitk_image,
            'image_path':image_path,
        }
        return  {k: v for k, v in return_dict.items() if v is not None}

    def get_mask_bbox(self, sitk_image, padding, image_path,objs_amount=2):
        lung_mask = mask.apply(sitk_image)
        np_mask = lung_mask  # z,y,x
        locs = ndimage.find_objects(np_mask)
        spacing = sitk_image.GetSpacing()[::-1]
        if isinstance(locs, list):
            if not len(locs) == objs_amount:
                logging.error('Can not find left or right lung: '+image_path)
            bbox = [[max(min([loc[zyx].start for loc in locs]) - ceil(padding / spacing[zyx]), 0),
                     min(max([loc[zyx].stop for loc in locs]) + ceil(padding / spacing[zyx]),
                         sitk_image.GetSize()[::-1][zyx])]
                    for zyx in range(0, 3)]
        else:
            logging.error('Can not find left or right lung: '+image_path)
            bbox=None
        return bbox, np_mask




if __name__ == '__main__':

    from importlib import reload
    logging.shutdown()
    reload(logging)

    logging.basicConfig(filename=join(save_path, 'run.log'), level=logging.DEBUG, filemode='a')
    logging.warning('Start logging')
    ds = Dataset_provider('ICCV_Lung')

    bbox_arr=[]
    save = True
    by_dir = False
    split_slice = True
    max_slice=100
    dict_train=[]
    dict_val=[]
    dict_test=[]
    with ThreadPoolExecutor(max_workers=6) as executor:
        for sub_ds in [ds.train,ds.val,ds.test]:
            if sub_ds:
                for i in dataloader(sub_ds,ds.dir_path,ds.type):
                    if save:
                        id = basename(i['image_path'])
                        if basename(dirname(i['image_path']))=='test':
                            split = basename(dirname(i['image_path']))
                        else:
                            cate = basename(dirname(i['image_path']))
                            split = basename(dirname(dirname(i['image_path'])))
                        if by_dir:
                            if split=='test':
                                img_path = join(save_path, split, id)
                            else:
                                img_path=join(save_path,split,cate,id)
                            os.makedirs(img_path, exist_ok=True)
                            executor.submit(sitk.WriteImage,i['sitk_im'],join(img_path,'image.nii.gz'))
                            executor.submit(sitk.WriteImage,i['sitk_lung_mask'], join(img_path, 'mask.nii.gz'))
                        else:
                            os.makedirs(join(save_path, 'data'), exist_ok=True)
                            if split == 'test':
                                img_header = join(save_path, 'data', split + '_' + id)
                            else:
                                img_header = join(save_path, 'data',split+'_'+cate+'_'+id)

                            [[zmin, zmax], [ymin, ymax], [xmin, xmax]] = i['bbox']
                            np_image= i['image'][zmin:zmax,ymin:ymax, xmin:xmax]
                            np_mask = i['lung_mask'][zmin:zmax,ymin:ymax, xmin:xmax]
                            if split_slice:
                                num_of_split=np_image.shape[0]//max_slice+1
                                if num_of_split==1:
                                    spacing=i['sitk_im'].GetSpacing()
                                    img_filename= img_header+'_0'
                                    sitk_image=sitk.GetImageFromArray(np_image)
                                    sitk_mask=sitk.GetImageFromArray(np_mask)
                                    sitk_image.SetSpacing(spacing)
                                    sitk_mask.SetSpacing(spacing)
                                    os.makedirs(img_filename, exist_ok=True)
                                    executor.submit(sitk.WriteImage, sitk_image, join(img_filename, 'masked_ct.nii.gz'))
                                    executor.submit(sitk.WriteImage, sitk_mask, join(img_filename, 'mask.nii.gz'))
                                    if split=='train':
                                        if type(i['label']) is int:
                                            dict_train.append({'case':basename(img_filename),'label':str(i['label'])})
                                        else:
                                            raise NotImplemented
                                    elif split=='val':
                                        if type(i['label']) is int:
                                            dict_val.append(
                                                {'case': basename(img_filename), 'label': str(i['label'])})
                                        else:
                                            raise NotImplemented
                                    elif split=='test':
                                        dict_test.append({'case': basename(img_filename)})

                                else:
                                    for n_split in range(0,num_of_split):
                                        np_image_sp=np_image[n_split::num_of_split]
                                        np_mask_sp=np_mask[n_split::num_of_split]
                                        spacing = i['sitk_im'].GetSpacing()
                                        spacing = (spacing[0],spacing[1],spacing[2]*num_of_split)
                                        img_filename = img_header + '_'+str(n_split+1)

                                        sitk_image = sitk.GetImageFromArray(np_image_sp)
                                        sitk_mask = sitk.GetImageFromArray(np_mask_sp)
                                        sitk_image.SetSpacing(spacing)
                                        sitk_mask.SetSpacing(spacing)
                                        os.makedirs(img_filename, exist_ok=True)
                                        executor.submit(sitk.WriteImage, sitk_image,
                                                        join(img_filename, 'masked_ct.nii.gz'))
                                        executor.submit(sitk.WriteImage, sitk_mask,
                                                        join(img_filename, 'mask.nii.gz'))
                                        if split == 'train':
                                            if type(i['label']) is int:
                                                dict_train.append(
                                                    {'case': basename(img_filename), 'label': str(i['label'])})
                                            else:
                                                raise NotImplemented
                                        elif split == 'val':
                                            if type(i['label']) is int:
                                                dict_val.append(
                                                    {'case': basename(img_filename), 'label': str(i['label'])})
                                            else:
                                                raise NotImplemented
                                        elif split == 'test':
                                            dict_test.append({'case': basename(img_filename)})

                            else:
                                raise NotImplemented
                            pass

                    csv_columns =['case','label']
                    with open(join(save_path,'train.csv'), 'w') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writeheader()
                        for data in dict_train:
                            writer.writerow(data)
                    with open(join(save_path,'val.csv'), 'w') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writeheader()
                        for data in dict_val:
                            writer.writerow(data)
                    with open(join(save_path,'test.csv'), 'w') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                        writer.writeheader()
                        for data in dict_test:
                            writer.writerow(data)
                    bbox={'path':i['image_path'].replace(ds.dir_path,''),'bbox':i['bbox']}
                    bbox_arr.append(bbox)
                    with open(join(save_path,'bbox.json'), 'w') as fp:
                        json.dump(bbox_arr, fp, indent=2)
