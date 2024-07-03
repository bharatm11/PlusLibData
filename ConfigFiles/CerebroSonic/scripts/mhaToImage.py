# This script reads a .mha sequence file, processes it to segment  out noise using simple 
# thresholding, and saves the edited frames into a new sequence.


import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import shutil

class MhaSlideshow:
    def __init__(self, file_path, interval=10):
        # Store the original .mha file path
        self.original_file_path = file_path
        # Read the .mha file
        self.image = sitk.ReadImage(file_path)
        # Convert the image to a numpy array
        self.image_array = sitk.GetArrayFromImage(self.image)
        # Get the number of frames
        self.num_frames = self.image_array.shape[0]
        self.current_frame = 0
        self.edited_image_array = np.copy(self.image_array)
        self.interval = interval

        # Process all frames
        self.process_all_frames()

        # Set up the plot with two subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.img1 = self.ax1.imshow(self.image_array[self.current_frame], cmap='gray')
        self.img2 = self.ax2.imshow(self.edited_image_array[self.current_frame], cmap='gray')
        self.ax1.set_title(f'Original Frame {self.current_frame + 1}')
        self.ax2.set_title(f'Edited Frame {self.current_frame + 1}')
        self.ax1.axis('off')
        self.ax2.axis('off')

        # Add a button for saving
        self.save_button_ax = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.save_button = Button(self.save_button_ax, 'Save')
        self.save_button.on_clicked(self.save_image)

        # Set up the animation
        self.ani = FuncAnimation(self.fig, self.update_frame, interval=self.interval, blit=False)
        
        plt.show()

    def process_all_frames(self):
        for i in range(self.num_frames):
            frame = self.image_array[i]

            # Convert to grayscale if not already
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = frame

            # Apply Gaussian smoothing
            smoothed_image = cv2.GaussianBlur(gray_image, (5, 5), 2)

            # Binarize the image using Otsu's thresholding
            _, binary_image = cv2.threshold(smoothed_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Detect edges using Sobel, Canny, and Prewitt methods
            sobel_edges = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 1, ksize=3)
            sobel_edges = np.uint8(np.absolute(sobel_edges))

            canny_edges = cv2.Canny(smoothed_image, 100, 200)

            prewitt_kx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            prewitt_ky = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            prewitt_edges_x = cv2.filter2D(smoothed_image, -1, prewitt_kx)
            prewitt_edges_y = cv2.filter2D(smoothed_image, -1, prewitt_ky)
            prewitt_edges = prewitt_edges_x + prewitt_edges_y

            # Combine edges
            combined_edges = np.bitwise_or(sobel_edges, canny_edges)
            combined_edges = np.bitwise_or(combined_edges, prewitt_edges)

            # Combine the binary image with the combined edges
            combined_image = np.bitwise_or(binary_image, combined_edges)

            # Mask the original grayscale image with the binary image
            masked_image = gray_image.copy()
            masked_image[binary_image == 0] = 0

            # Update the edited image array
            self.edited_image_array[i] = masked_image

    def update_frame(self, *args):
        self.current_frame = (self.current_frame + 1) % self.num_frames
        self.img1.set_data(self.image_array[self.current_frame])
        self.img2.set_data(self.edited_image_array[self.current_frame])
        self.ax1.set_title(f'Original Frame {self.current_frame + 1}')
        self.ax2.set_title(f'Edited Frame {self.current_frame + 1}')
        self.fig.canvas.draw_idle()

    def save_image(self, event):
        # Create a new file name for the edited .mha file
        edited_file_path = self.original_file_path.replace('.mha', '_edited.mha')

        # Convert the numpy array back to a SimpleITK image
        new_image = sitk.GetImageFromArray(self.edited_image_array)
        # Copy all metadata from the original image to the new image
        for key in self.image.GetMetaDataKeys():
            new_image.SetMetaData(key, self.image.GetMetaData(key))
        
        # Save the new image to the edited file path
        sitk.WriteImage(new_image, edited_file_path)
        print(f'Saved edited image to {edited_file_path}')

# Example usage
file_path = r'seq_UT_Bharat_Bilateral_Falx_Bmode.mha'
MhaSlideshow(file_path, interval=5)  # Set interval to 10 milliseconds
