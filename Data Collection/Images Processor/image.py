import cv2
import time
import threading
import numpy as np
import csv

# Global variables
path = "samples/sample6.jpg"
start_x, start_y = -1, -1
end_x, end_y = -1, -1
average_rgb = None
thick = 1


# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    global start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        end_x, end_y = x, y
        if end_x < start_x:
            aux = start_x
            start_x = end_x
            end_x = aux
        if end_y < start_y:
            aux = start_y
            start_y = end_y
            end_y = aux
        draw_rectangle()
        threading.Thread(target=clear_rectangle_after_delay).start()
        threading.Thread(target=calculate_average_rgb).start()


# Function to draw the rectangle
def draw_rectangle():
    global image, start_x, start_y, end_x, end_y
    if start_x != -1 and start_y != -1 and end_x != -1 and end_y != -1:
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), thick)
        cv2.imshow("Image", image)
        print(f"Selected from ({start_x},{start_y}) to ({end_x},{end_y})")


# Function to clear the rectangle after a delay
def clear_rectangle_after_delay():
    time.sleep(2)  # Delay for 2 seconds
    clear_rectangle()


# Function to clear the rectangle
def clear_rectangle():
    global image
    image = cv2.imread(path)  # Reload the original image
    cv2.imshow("Image", image)


# Function to calculate the average RGB values
def calculate_average_rgb():
    global image, start_x, start_y, end_x, end_y, average_rgb
    if start_x != -1 and start_y != -1 and end_x != -1 and end_y != -1:
        masked_image = image[start_y+thick:end_y-thick, start_x+thick:end_x-thick]
        if masked_image.size > 0:
            average_rgb = np.mean(masked_image, axis=(0, 1))
            print("Average RGB values:", average_rgb[2], average_rgb[1], average_rgb[0])
            save_to_csv()


# Function to save data to a CSV file
def save_to_csv():
    global average_rgb
    if average_rgb is not None:
        number = input("Enter a number: ")
        filename = "data.csv"
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([number, average_rgb[2], average_rgb[1], average_rgb[0]])


# Load an image
image = cv2.imread(path)  # Replace with the path to your image

# Resize the image to fit the window properly
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)

# Set the mouse callback function
cv2.setMouseCallback("Image", mouse_callback)

cv2.waitKey(0)
cv2.destroyAllWindows()
