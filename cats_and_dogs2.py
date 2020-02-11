import matplotlib.pyplot as plt
import numpy as np
import cv2
import tflearn

im_size = 50

def Load_model():
    model = tflearn.DNN()
    new = model.load('dogsVscats- 0.001-2conv-basic-video.model')
    print("Loaded model!")
    return new
def Image(new):
    fig=plt.figure()
    img_array = cv2.imread("dog.jpg", cv2.IMREAD_GRAYSCALE)
    data = cv2.resize(img_array,(im_size,im_size))
    x = []
    x.append(data)
    x = np.asarray(x)
    for i in range(0,2):
        x = np.moveaxis(x, -1, 0)
        
    
    model_out = new.predict([x])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
            
    plt.imshow(data)
    plt.title(str_label)
    plt.show()
    
if __name__ == '__main__':
    new = Load_model()
    Image()


