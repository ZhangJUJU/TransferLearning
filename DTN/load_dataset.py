import numpy as np
import tensorflow as tf

def read_date(filenum):
    # 0:mnist 1:usps
    if filenum == 'mnist-fake':
        date_from_file = np.load('mnist16-fake.npy')
        print("loaded mnist16", date_from_file.shape)
    elif filenum == 'usps':
        date_from_file = np.load('usps9298-shuffled.npy')
        print("loaded usps16",date_from_file.shape)
    elif filenum == 'mnist-real':
        date_from_file = np.load('mnist16-shuffled.npy')
        print("loaded mnist16", date_from_file.shape)
    else:
        print("import filenum input error! will return")
    labels=np.int32(date_from_file[:,0])
    labels=dense_to_one_hot(labels,10)
    images=date_from_file[:, 1:]
    if filenum == 'usps':
        images=images/np.float32(2)
    if filenum == 'mnist-real':
        images=images/np.float32(256)


    return images, labels
def get_minibatch(images,labels,BATCH_SIZE):

    Total=np.hstack([labels,images])
    np.random.shuffle(Total)
    batch_image=Total[:np.int32(BATCH_SIZE/2),10:]
    batch_label=Total[:np.int32(BATCH_SIZE/2),:10]
    return batch_image,batch_label
def dense_to_one_hot(labels_dense, num_classes):

  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + np.int32(labels_dense).ravel()] = 1
  print("to one done")
  return labels_one_hot

if __name__ == '__main__':
    read_date('mnist-fake')


