import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.preprocessing import MinMaxScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import glob2
import sys
sys.path.append("C:/xgboost/python-package")
import xgboost as xgb
from scipy.ndimage.measurements import label


orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2
hog_channel = 1
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy



    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.7, 0.7)):
    
    # If x and/or y start/stop positions not defined, set to image size
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]
        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan/nx_pix_per_step) - 1
        ny_windows = np.int(yspan/ny_pix_per_step) - 1
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        # Note: you could vectorize this step, but in practice
        # you'll be considering windows one by one with your
        # classifier, so looping makes sense
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list
    
    
    
def single_extract(image, cspace="YCrCb", 
                   orient=9, pix_per_cell=8, cell_per_block=2, feature_vec=True):
    # Create a list to append feature vectors to
        features = []

        # Iterate through the list of images
        # Read in each one by one
        # apply color conversion if other than 'RGB'
        
        image = cv2.resize(image, (64, 64))
        
        feature_image = newSpace(image)
        # Call get_hog_features() with vis=False, feature_vec=True
        #hog_features = []
        #for channel in range(feature_image.shape[2]):
            #hog_features.append(get_hog_features(feature_image[:,:,channel], 
             #                   orient, pix_per_cell, cell_per_block, 
              #                  vis=False, feature_vec=feature_vec))
            
        hog_features = get_hog_features(feature_image[:,:,1], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=feature_vec)
        hog_features = np.ravel(hog_features)        

        hog_features = hog_features.reshape(1, -1)

        #Combine in order of hist, bin and hog
        
        return hog_features
    
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

    
        
        
def newSpace(image):
    R = scaler.fit_transform(image[:, :, 0])
    #B = scaler.fit_transform(image[:, :, 2])
    Y = scaler.fit_transform(cspace(image, color_space='YCrCb')[:, :, 0])
    L = scaler.fit_transform(cspace(image, color_space='HLS')[:, :, 0])
    return np.dstack((R, Y, L))
    
    
def cspace(image, color_space='RGB'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(image)    
    return feature_image

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def search_windows(img, windows, clf, color_space='RGB', 
                    orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))   
        
        
        
        #4) Extract features for that window using single_img_features()
        features = single_extract(test_img)
        #5) Scale extracted features to be fed to classifier
        #test_features = scaler.transform(np.array(features).reshape(1, -1))
        #test_features = np.array(features).reshape(1, -1)
        #6) Predict using your classifier
        prediction = clf.predict(xgb.DMatrix(features))
        #7) If positive (prediction == 1) then save the window
        
        ## For xgboost, using 0.4 as threshold
        prediction = (prediction > 0.9) * 1
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows






def box(image, bst):
    windows = slide_window(image, x_start_stop=[800, None], xy_window=(120, 120), y_start_stop=[400, 656], xy_overlap=(0.7, 0.7))
    #windows2 = slide_window(image, x_start_stop=[64*11, None], xy_window=(80, 80), y_start_stop=[int(image.shape[0]/2), int(image.shape[0])], xy_overlap=(0.9, 0.9))
    #windows3 = slide_window(image, x_start_stop=[800, None], xy_window=(96, 96), y_start_stop=[400, 656], xy_overlap=(0.8, 0.8))

    #windows.extend(windows2)
    #windows.extend(windows3)

    hot_windows = search_windows(image, windows, bst,  
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel)                       

    window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)  
    
    return window_img, hot_windows


def Heat(image, bst):
    result, hot_windows = box(image, bst)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    temp = add_heat(heat, hot_windows)
    tempi = apply_threshold(temp, 0)
    return temp


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (60,15,180), 6)
    # Return the image
    return img

    
    
def Label(image, bst):
    result, hot_windows = box(image, bst)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    temp = add_heat(heat, hot_windows)
    tempi = apply_threshold(temp, 0)
    labels = label(tempi)
    
    return draw_labeled_bboxes(image, labels)
    