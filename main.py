from api import Face
import argparse

"""
modes here shows how you want to use the kit it can be

'detect'
'vid_face_detect'

"""
def main(args):
    face_dir=args.fd
    vid_dir=args.vd
    mode=args.m
    face=Face(mode,face_dir,vid_dir)
    face.start()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='a kit in ordert to handle faces in images or in a video')
    parser.add_argument('-m',type=str,default='detect',help='set the mode of the facekit \n detect for face detection in images from a folder \n '
                                                            'vid_face_detect for to detect faces in an vidoe and save them in a folder'
                                                            'vide_face_track to track faces from an object')
    parser.add_argument('-fd',type=str,help='sets faces path',)
    parser.add_argument('-vd', type=str,help='sets video path')

    args = parser.parse_args()
    main(args)