import cv2
from picamera2 import Picamera2

cam = Picamera2()
cam.preview_configuration.main.size = (800, 600)
cam.preview_configuration.main.format= "RGB888"
cam.preview_configuration.align()
cam.configure("preview")
cam.start()

width, height = 800, 600
# 

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

while True:
	frame = cam.capture_array()
	cv2.imshow("PicamNoIR", frame)
	
	out.write(frame)
	
	if cv2.waitKey(1)==ord('q'):
		break

cv2.destroyAllWindows()
out.release()
