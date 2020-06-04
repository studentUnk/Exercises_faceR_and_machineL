# Exercise of face recognition
This exercise is part of a project to be implemented in the university *Minuto de Dios*.

The main idea on this exercise is to implement in the most basic way some of algorithms of face recognition to set a layer of security. To achieve this the library of OpenCV is being used and also the repository of face_recognition (ageitgey).

The process to save the images, as as it is written in the recommendations in OpenCV, is create a general folder and identify all the images of a subject in a single folder, in this way is easier to identify each person. 

I am not the owner of most of the code that is written here, so, below there are some of the main references that are the base for this code:

**Database's faces:**
- [Yale Face Database](http://vision.ucsd.edu/content/yale-face-database): *Test recognition with open faces*
**Handle the images**
- [OpenCV](https://docs.opencv.org/3.4/): *It does all the work of find a face, train with a dataset and recognizing (eigenface, fisherface, LBPH) *
- [cvlib](https://github.com/tanay-bits/cvlib/tree/master/Face%20Recognition): *Repository with an stable (but old) implementation of face recognition using opencv*
- [face_recognition](https://github.com/ageitgey/face_recognition): *Train and recognize faces of adults*
