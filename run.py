
import cv2 as cv

path = '/train/train_0.jpg'
image = cv.imread(path)
mask, sudoku_digits = predict_image(image)
