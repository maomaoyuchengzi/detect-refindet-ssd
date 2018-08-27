import numpy as np  
import sys,os  
import cv2
caffe_root = '/home/fuxueping/sdb/Caffe_Project_Train/caffe_refinedet/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  


net_file= '/home/fuxueping/sdb/temp/l1_pruned_deploy.prototxt'
caffe_model='/home/fuxueping/sdb/temp/146174_refinedet_l1ranked_320x320_iter_80000.caffemodel'
test_dir = "/home/fuxueping/sdb/PycharmProjects/TensorRT/refindet/test_img"
# caffe.set_mode_cpu()

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()


CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')



def preprocess(src):
    img = cv2.resize(src, (320,320))
    # img = img - 127.5
    # img = img * 0.007843#0.007843 = 1/127.5

    b, g, r = cv2.split(img)
    b = b - 104.0
    g = g - 117.0
    r = r - 123.0

    img = cv2.merge((b, g, r))

    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    # np.savetxt("/home/yanggui/Downloads/MobileNet-SSD-master/fname.txt",out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h]))
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    net = caffe.Net(net_file, caffe_model, caffe.TEST)
    origimg1 = cv2.imread(imgfile) #BGR
    type = origimg1.dtype #type = uint8
    origimg = origimg1.astype(np.float32)#转换数据类型
    img = preprocess(origimg)

    img = img.transpose((2, 0, 1))

    f = open("/home/fuxueping/sdb/PycharmProjects/TensorRT/input_data.txt", 'w')
    channel_count = img.shape[0]
    height = img.shape[1]
    width = img.shape[2]
    for c in range(channel_count):
        for h in range(height):
            for w in range(width):
                data = img[c][h][w]
                f.write('%s\n' % str(data))
                # print(data)
    f.close()


    #tmp = [[0.2,0.3 ] ,[0.2 ,0.4]]
    #tmp1 = np.array(tmp).reshape((1,2,2,1))


    net.blobs['data'].data[...] = img

    out = net.forward()


    f = open("/home/fuxueping/sdb/PycharmProjects/TensorRT/out_put/detection_out.txt", 'w')
    layername = "detection_out"
    if layername in net.blobs:
        # feature_map = self.blobs_data(layername)
        feature_map = net.blobs[layername]
        height = feature_map.height
        width = feature_map.width
        channels = feature_map.channels

        for c in range(channels):
            for h in range(height):
                for w in range(width):

                    data1 = feature_map.data[0][c][h][w]
                    f.write('%f\n'%( data1) )
        f.close()

    box, conf, cls = postprocess(origimg, out)
    #np.savetxt("/home/yanggui/Downloads/MobileNet-SSD-master/fname.txt",out['detection_out'][0,0,:,:])
    
    

    for i in range(len(box)):
       p1 = (box[i][0], box[i][1])
       p2 = (box[i][2], box[i][3])   
       cv2.rectangle(origimg1, p1, p2, (0,255,0))
       p3 = (max(p1[0], 15), max(p1[1], 15))
       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
       cv2.putText(origimg1, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
    # cv2.imshow("SSD", origimg)
    cv2.imwrite("1.jpg",origimg1)
 
    # k = cv2.waitKey(0) & 0xff
    #     #Exit if ESC pressed
    # if k == 27 : return False
    return True

for f in os.listdir(test_dir):
    if detect(test_dir + "/" + f) == False:
       break
