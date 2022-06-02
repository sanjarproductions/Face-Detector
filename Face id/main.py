import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('brin.jpg')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinares = trained_face_data.detectMultiScale(grayscaled_img) 

for (x,y,w,h) in face_coordinares:
    cv2.rectangle(img, (x,y), (x+w,y+h), ((0, 255, 0)),4)

cv2.imshow('Sanjar Productions', img)

print(face_coordinares) 

cv2.waitKey()

print("Code Complete")

# import cv2
# from random import randrange

# trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('ME.jpg')
 
# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# for (x,y,w,h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h),(randrange(256), randrange(256), randrange(256)), 4)


# cv2.imshow('Clever Programmer Face Detection', img)

# print(face_coordinates) 

# cv2.waitKey()

# print("Code Complete")