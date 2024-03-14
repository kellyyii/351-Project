<img width="483" alt="image" src="https://github.com/kellyyii/351-Project/assets/71577249/570b11ed-4438-41e0-928c-6eda52ba5457">


Big thanks to How to do Facial Recognition And Overlay with Image Asset (Python2 & OpenCV: png with Alpha Channel Transparency) by Blade Nelson
Link: https://www.codementor.io/@powderblock/how-to-do-facial-recognition-and-overlay-superimpose-image-assets-python2-opencv-png-with-alpha-channel-transparency-zeqf7rjh0

Biggest Bugs:

1. Cannot goes transparent
<img width="490" alt="image" src="https://github.com/kellyyii/351-Project/assets/71577249/1135e622-71c9-4786-a72a-43a3c361d514">

2. numpy.core._exceptions._UFuncOutputCastingError:
Cannot cast ufunc 'add' output from dtype('float64') to dtype('uint8') with casting rule 'same_kind'

The data types of the headdress image (hood_resized) and "alpha_mask" are both floating point numbers (float32), while the data type of "background(bg)" is an unsigned integer (uint8). When using the np.add function, the output data type must be the same as the data type of bg, so type conversion is required.

solve: converting bg's data type to float (float32).

3. Error code 0011
<img width="484" alt="image" src="https://github.com/kellyyii/351-Project/assets/71577249/7605a9b5-c903-458a-aa3b-c021d87a328d">

