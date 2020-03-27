# Face Kit (Detecting,Clustering,Tracking)

Face recognition toolkit using face-recognition library
this project is a toolkit in order to detect,cluster and track faces in a a video or bunch of images
## Getting Started

In this project we've used face_recognition library and dlib to recognize faces 

### Prerequisites
To run this project you need:
- python 3
- [face_recognition](https://github.com/ageitgey/face_recognition) 
- dlib



To install dlib in Ubunto i've used [this](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf) installation guide but just to make it more simpler here are the steps:



The requirements are stored in requirment.txt which you can install using the code below

```
pip install -r requirment.txt
```

If dlib is not installed properly on your device you can use the below code obtained from [here](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf)

```
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build; cd build; cmake ..; cmake --build .
cd ..
python3 setup.py install
```
## Running the tests

In order to test this project you can use command line tools. To do so you can run main script with following arguments

```
python main.py -h
```

The -h arguments shows a brief help on the program

There are several other arguments which help you run the script 
- -m : this arguments shows the mode in which you want to run the program
- -fd: this argument is to set the directory which contains of pictures you want to detect faces in them
- -vd: this argument sets the vidoe path you want to process
### Detection

To run the programm in Detection mode the -m argument should be set to detect and the -fd should be set by the folder which contains your images
the output is a text file which contains a dictionary of the location of faces in images

```
python  main.py -m detection -fd faces
```

### Video face clustering 

If you want to run to cluster the faces in a video you should set the mode to vid_face_detect and set -vd to the video path
the results are faces croped from the video and the face for each person is stored in a seperate folder 

```
python  main.py -m vid_face_detect -vd 1.mp4
```
### Video face tracking 

If you want to track all the faces in a video and assign a unique id to them you should set the mode to vid_face_track and again -fv to the video path as below:

```
python  main.py -m vid_face_track -vd 1.mp4
```

The output is a video which each face is described by a unique id

The output is like the gif below

![Recordit GIF](output.gif)
## Author

* **Fateme Bafghi** 
