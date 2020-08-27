from tensorflow.keras import models 
import sys
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.cm as mtpltcm

def main(argv):
    cap = cv2.VideoCapture(0)
    model=models.load_model('hand-model.h5')
    
    rev={0: '01_palm', 1: '02_l', 2: '03_fist', 3: '04_fist_moved', 4: '05_thumb', 5: '06_index', 6: '07_ok',
 7: '08_palm_moved', 8: '09_c', 9: '10_down'}

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        #Resizing frame
        colors = cv2.resize(frame, (320,120))
        colors=cv2.cvtColor(colors, cv2.COLOR_BGR2GRAY)
        colors=np.array(colors)
        colors=colors/255
        colors=colors.reshape((120,320,1))
        
        colors=np.expand_dims(colors,axis=0)      
        
        print(rev[np.argmax(model.predict(colors))])
        
        cv2.imshow('frame',colors[0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))