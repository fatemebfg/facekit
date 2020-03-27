from PIL import Image
import face_recognition
import os
from sklearn.cluster import DBSCAN
import numpy as np
import pickle
from logzero import logger
import logzero
import cv2


class Face:
    def __init__(self,mode=None,face_dir=None,video_dir=None):
        self.face_dir=face_dir
        self.mode=mode
        self.video_dir=video_dir


    def start(self):
        function = getattr(self, self.mode)
        results=function()
        return results

    def batchdetect(self,face_dir=None):
        pictures = []
        if face_dir is None:
            face_dir=self.face_dir
        for file in sorted(os.listdir(face_dir)):
            image = face_recognition.load_image_file(os.path.join(face_dir, file))
            location = face_recognition.face_locations(image,model='hog')
            encoding = face_recognition.face_encodings(image,known_face_locations=location)
            pictures.append(dict(name=file, location=location, encoding=encoding, image=image))

        f = open('face_location', "wb")
        f.write(pickle.dumps(pictures))
        f.close()
        self.faces=pictures
        return(pictures)

    def detect(self,file_path=None,image=None):

        if image is None:
            image = face_recognition.load_image_file(file_path)
        location = face_recognition.face_locations(image,model='hog')
        return(location)

    def cluster(self,from_file=True,file_path='face_location'):
        if not os.path.exists('clustering'):
            os.makedirs('clustering')

        if from_file==True:
            try:
                data = pickle.loads(open(file_path, "rb").read())
            except:
                return 'file not foound'
        else:
            data=self.faces

        encodings = list()
        for d in data:
            encodings.extend(d["encoding"])

        encodings = np.array(encodings)
        clt = DBSCAN(min_samples=1, metric="euclidean",eps=0.4)
        clt.fit(encodings)
        i=0
        for img in data:
            image = Image.fromarray(img['image'])
            for loc in img['location']:
                if not os.path.exists(os.path.join('clustering',str(clt.labels_[i]))):
                    os.makedirs(os.path.join('clustering',str(clt.labels_[i])))
                top, right, bottom, left = loc
                crop = image.crop((left, top, right, bottom))
                crop.save(os.path.join('clustering',str(clt.labels_[i]),str(clt.labels_[i]) +'_'+str(i)), 'png')
                i+=1

    def vid_face_detect(self,file=None,cluster=True):
        if file==None:
            file=self.video_dir
        input_movie=cv2.VideoCapture(file)


        frame_number=0
        pictures=[]
        logger.info('video analysig started')
        while True:
            ret,frame = input_movie.read()
            frame_number+=1

            if not ret:
                break

            rgb_frame=frame[ :, :, ::-1]
            location=face_recognition.face_locations(rgb_frame,model='cnn')
            logger.info('frame no {} has {} faces'.format(frame_number,len(location)))
            if len(location)>0:
                encoding=face_recognition.face_encodings(rgb_frame,known_face_locations=location)
                pictures.append(dict(name=file, location=location, encoding=encoding, image=frame))
            logzero.logfile('testlog.txt', maxBytes=1e6, backupCount=5)
        f = open('face_location', "wb")
        f.write(pickle.dumps(pictures))
        f.close()
        if cluster==True:
            self.cluster()
        self.faces = pictures
        return (pictures)

    def vid_face_track(self,file=None):
        if file==None:
            file=self.video_dir
        input_movie=cv2.VideoCapture(file)
        input_movie.set(cv2.CAP_PROP_FPS,3)
        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_movie = cv2.VideoWriter('output.avi', fourcc, 29.97, (640, 360))

        frame_number=0
        pictures=[]
        logger.info('video analysing started')
        known_face_encodings=[]
        trace=dict()
        best_match_index=[]
        while True:
            ret,frame = input_movie.read()
            frame_number+=1

            if not ret:
                break

            rgb_frame=frame[ :, :, ::-1]
            location=face_recognition.face_locations(rgb_frame,model='cnn')
            logger.info('frame no {} has {} faces'.format(frame_number,len(location)))
            if len(location)>0 :
                encoding=face_recognition.face_encodings(rgb_frame,known_face_locations=location)
                if len(known_face_encodings) == 0:
                    known_face_encodings.extend(encoding)
                    for i,e in enumerate(encoding):
                        trace.update({i:[i]})

                else:


                    face_distances =self.check_faces(known_face_encodings, encoding,trace)
                    best_match_index =[ np.argmin(d) for d in face_distances]
                    ids=[]
                    for indx,matching in enumerate(best_match_index):
                        if face_distances[indx][matching]>0.6:
                            key=max(trace.keys())+1
                            trace.update({key: [len(known_face_encodings)+indx]})
                            ids.append(key)
                        else:
                            ids.append(best_match_index[indx])
                    known_face_encodings.extend(encoding)


                    for (top, right, bottom, left), name in zip(location, ids):
                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, str(name), (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

                # Write the resulting image to the output video file
            print("Writing frame {} / {}".format(frame_number, length))
            output_movie.write(frame)
        input_movie.release()
        cv2.destroyAllWindows()



        logzero.logfile('testlog', maxBytes=1e6, backupCount=5)

        f = open('face_location', "wb")
        f.write(pickle.dumps(pictures))
        f.close()
        self.faces = pictures
        return (pictures)


    def check_faces(self,known_face_encodings,face_encodings,trace):
        #checks a list of face encodings with a simple known_face_encoding
        matches=[]
        for face in face_encodings:
            face_matches=[]
            for key,val in trace.items():
                avg=np.mean(face_recognition.face_distance([known_face_encodings[item] for item in val],face))
                face_matches.append(avg)
            matches.append(face_matches)
        return matches

















