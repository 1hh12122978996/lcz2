# 为原始数据集增加新的波段，如ndvi
import numpy as np
import h5py
import random
import time


# def load_mosaic(self,index):
#     """将四张相同label的图像拼在一张马赛克图像中
#     index：需要获取的图像索引"""
#     indices = []
#     img1 = np.hstack((image1,image2))
#     img2 = np.hstack((image3,image4))
#     img = np.stack((img1,img2))

def stack_h5():
    archive = h5py.File(r'../../../so2sat/training.h5', 'r')
    # archive = h5py.File(r'../../../so2sat/testing.h5', 'r')
    # archive = h5py.File(r'G:\lcz\m1483140\m1483140/testing.h5', 'r')
    # archive = h5py.File(r'G:\lcz\m1483140\m1483140/testing.h5', 'r')
    data = archive['sen2']
    labels = archive['label']
    print(data.shape,labels.shape)

    data_new = np.empty((0,64,64,10))
    labels_new = np.empty((0,17))

    for i in range(17):
        index = np.where( np.argmax(labels,axis=1) == i )
        print(len(index[0]))
        for j in index[0]:
            indices = [j] + [j+1,j+2,j+3]
            img1 = np.hstack((data[indices[0]], data[indices[1]]))
            img2 = np.hstack((data[indices[2]],data[indices[3]]))
            img = np.vstack((img1,img2))

            img = np.expand_dims(img,axis=0)
            label = np.expand_dims(labels[j], axis=0)
            data_new = np.concatenate((data_new,img),axis=0)
            labels_new = np.concatenate((labels_new,label),axis=0)

    print(data_new.shape)
    print(labels_new.shape)

    h5_stack = h5py.File(r'../../../so2sat/train_stack.h5', 'w')
    # h5_stack = h5py.File(r'../../../so2sat/test_stack.h5', 'w')
    # h5_stack = h5py.File(r'G:\lcz\m1483140\m1483140/test_stack.h5', 'w')
    # h5_stack = h5py.File(r'G:\lcz\m1483140\m1483140/train_stack.h5', 'w')
    h5_stack.create_dataset('sen2',data=data_new)
    h5_stack.create_dataset('label',data=labels_new)

t1 = time.time()
stack_h5()
t2 = time.time()
print(t2-t1)

#
# archive =h5py.File(r'G:\lcz\m1483140\m1483140/testing.h5', 'r')
# data = archive['sen2']
# labels = archive['label']
#
# index = 1
#
# image = data[index]
# image = np.transpose(image,(2,0,1))
# mask = labels[index]
# print(mask, np.argmax(mask, axis=0),np.argmax(labels, axis=1))
#
# index_list = np.where(np.argmax(labels, axis=1) == np.argmax(mask, axis=0))
# print(len(index_list[0]))
#
# indices = [index] + list(np.random.choice(index_list[0], 3))
# print(data[indices[0]].shape)
# img1 = np.hstack((data[indices[0]], data[indices[1]]))
# img2 = np.hstack((data[indices[2]], data[indices[3]]))
# image3 = np.concatenate((data[indices[0]], data[indices[1]]),axis=0)
# image4 = np.concatenate((data[indices[2]], data[indices[3]]),axis=0)
# image5 = np.concatenate((image3, image4),axis=1)
# print('image3',image5.shape)
# img = np.vstack((img1, img2))
#
# print('image形状',img.shape,image.shape,mask.shape)

