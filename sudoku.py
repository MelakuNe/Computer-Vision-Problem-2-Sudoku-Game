import numpy as np
import cv2 as cv
import cv2
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K




def predict_image(image: np.ndarray) -> (np.ndarray, list):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    sudoku_digits = [predict_sudoku(image)]
    
    mask = np.bool_(np.ones_like(image))
    _, mask = good_one(image)
                    
    return mask, sudoku_digits

def calc(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def get_sudoku_out(img, crop_rect):
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([calc(bottom_right, top_right), calc(top_left, bottom_left),
    calc(bottom_right, bottom_left), calc(top_left, top_right)])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    m = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, m, (int(side), int(side)))


def make_grid(img):
    squares = []
    side = img.shape[:1]
    side = side[0] / 9

    for j in range(9):
        for i in range(9):
            p1 = (i * side, j * side)  
            p2 = ((i + 1) * side, (j + 1) * side) 
            squares.append((p1, p2))
    return squares


def extract_from_square(img, rect):
    return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

def optimize(img, size, margin=0, background=0):
    h, w = img.shape[:2]

    def centre_pad(length):
        if length % 2 == 0:
            side1 = int((size - length) / 2)
            side2 = side1
        else:
            side1 = int((size - length) / 2)
            side2 = side1 + 1
        return side1, side2

    def scale(r, x):
        return int(r * x)

    if h > w:
        t_pad = int(margin / 2)
        b_pad = t_pad
        ratio = (size - margin) / h
        w, h = scale(ratio, w), scale(ratio, h)
        l_pad, r_pad = centre_pad(w)
    else:
        l_pad = int(margin / 2)
        r_pad = l_pad
        ratio = (size - margin) / w
        w, h = scale(ratio, w), scale(ratio, h)
        t_pad, b_pad = centre_pad(h)

    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))


def fill_cell(inp_img, scan_tl=None, scan_br=None):

    img = inp_img.copy()  # Copy the image, leaving the original untouched
    height, width = img.shape[:2]

    max_area = 0
    seed_point = (None, None)

    if scan_tl is None:
        scan_tl = [0, 0]

    if scan_br is None:
        scan_br = [width, height]

    for x in range(scan_tl[0], scan_br[0]):
        for y in range(scan_tl[1], scan_br[1]):
            if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                area = cv2.floodFill(img, None, (x, y), 64)
                if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                    max_area = area[0]
                    seed_point = (x, y)

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 255 and x < width and y < height:
                cv2.floodFill(img, None, (x, y), 64)

    mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

    if all([p is not None for p in seed_point]):
        cv2.floodFill(img, mask, seed_point, 255)

    top, bottom, left, right = height, 0, width, 0

    for x in range(width):
        for y in range(height):
            if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                cv2.floodFill(img, mask, (x, y), 0)

            if img.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right

    bbox = [[left, top], [right, bottom]]
    return img, np.array(bbox, dtype='float32'), seed_point


def find_digit_from_sudoku(img, rect, size):
    digit = extract_from_square(img, rect)  # Get the digit box from the whole square
    h, w = digit.shape[:2]
    margin = int(np.mean([h, w]) / 2.5)
    _, bbox, seed = fill_cell(digit, [margin, margin], [w - margin, h - margin])
    digit = extract_from_square(digit, bbox)

    w = bbox[1][0] - bbox[0][0]
    h = bbox[1][1] - bbox[0][1]

    if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
        return optimize(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)


def digits_from_cell(img, squares, size):
    digits = []
    _, img = filter_and_contour(img.copy())

    for square in squares:
        digits.append(find_digit_from_sudoku(img, square, size))
    return digits


def filter_and_contour(gray):
    blur = cv.GaussianBlur(gray, (15, 15), 0) 
    threshold_img = cv2.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    threshold_img = cv2.bitwise_not(threshold_img, threshold_img)

    contours, hierarchy = cv.findContours(threshold_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours, threshold_img

def finder(lis, index):
    g = sorted(lis)[index]
    for ix, j in enumerate(lis):
        if g == j:
            iinn = ix        
    return iinn

def get_get(h1, h2, polygon, i):
    br = finder(h1, -i)
    tl = finder(h1, i-1)
    bl = finder(h2, i-1)
    tr = finder(h2, -i)
    aa = [polygon[tl][0], polygon[tr][0], polygon[br][0], polygon[bl][0]]
    aa2 = np.array(aa).reshape(4, 2)
    return aa, aa2

# method 3 to find the corners coordinate
def good_one(image):
    contours, _ = filter_and_contour(image)
    connnt = sorted(contours, key=cv.contourArea, reverse=True)
    polygon = connnt[0]
    h1 = []
    h2 = []
    for idx, pt  in enumerate(polygon):
        h1.append([pt[0][0] + pt[0][1]])
        h2.append([pt[0][0] - pt[0][1]])
        
    cont_first,  cont_first2 = get_get(h1, h2, polygon, 1)
    tofill = np.zeros(image.shape)
    filled = cv.fillPoly(tofill, [cont_first2], (1, 1, 1))
    return cont_first, filled

def show_digits(digits, colour=255):
    rows = []
    with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
    for i in range(9):
        row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
        rows.append(row)
    img = np.concatenate(rows)
    return img

def pre_test(sudoku, model):
    sudoku = cv.resize(sudoku, (450,450))
    grid = np.zeros([9,9])
    for i in range(9):
        for j in range(9):
            image = sudoku[i*50:(i+1)*50, j*50:(j+1)*50]
            if image.sum() > 25000:    
                find_out = predict_each(image, model)
                if find_out != 1:
                    grid[i][j] = find_out
                else:
                    grid[i][j] = -1
            else:
                grid[i][j] = -1    
    grid =  grid.astype(int)
    return grid


def predict_each(image, model):
    image = cv.resize(image, (28,28))  
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image, verbose = 0)
    final_prediction = np.argmax(prediction)
    return final_prediction


def predict_sudoku(image):
    get_model = load_model('/autograder/submission/sudoku_model.h5')
    corners, _ = good_one(image)
    cropped = get_sudoku_out(image, corners)    
   
    squares = make_grid(cropped)
    digits = digits_from_cell(cropped, squares, 28)
    last_item = show_digits(digits)
    predicted_digits = pre_test(last_item, get_model)
    return predicted_digits



'''
# mnist model building
np.random.seed(42)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') 
x_train = x_train / 255
x_test = x_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# build the model
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# model.summary()


# compiling the model
# model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=200)

model.save("sudoku_model.h5")
'''