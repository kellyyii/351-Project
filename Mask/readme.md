<img width="478" alt="image" src="https://github.com/kellyyii/351-Project/assets/71577249/62ccaf10-983e-4fc2-8fca-7ced349cc970">
<img width="485" alt="image" src="https://github.com/kellyyii/351-Project/assets/71577249/8b795d4d-4024-4ca3-9235-2499f74e5c80">
<img width="484" alt="image" src="https://github.com/kellyyii/351-Project/assets/71577249/4ec0fc11-0cac-479d-99f0-2657308a9315">





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

4. IndexError
np.multiply(bg, np.atleast_3d(255 - hood_resized[:,:,3])/255.0,out=bg, casting="unsafe")
                                        ~~~~~~~~~~~~^^^^^^^
IndexError: index 3 is out of bounds for axis 2 with size 3

Solve: remove [:,:,3]



