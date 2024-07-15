import SimpleITK as sitk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

class MhaSlideshow:
    def __init__(self, file_path, interval=10):
        self.original_file_path = file_path
        self.image = sitk.ReadImage(file_path)
        self.image_array = sitk.GetArrayFromImage(self.image)
        self.num_frames = self.image_array.shape[0]
        self.current_frame = 0
        self.edited_image_array = np.copy(self.image_array)
        self.interval = interval
        self.paused = False
        self.points = []

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.img1 = self.ax1.imshow(self.image_array[self.current_frame], cmap='gray')
        self.img2 = self.ax2.imshow(self.edited_image_array[self.current_frame], cmap='gray')
        self.ax1.set_title('Select origin and two radius points for the fan shape region')
        self.ax2.set_title(f'Edited Frame {self.current_frame + 1}')
        self.ax1.axis('off')
        self.ax2.axis('off')

        self.save_button_ax = self.fig.add_axes([0.81, 0.05, 0.1, 0.075])
        self.save_button = Button(self.save_button_ax, 'Save')
        self.save_button.on_clicked(self.save_image)

        self.pause_button_ax = self.fig.add_axes([0.81, 0.15, 0.1, 0.075])
        self.pause_button = Button(self.pause_button_ax, 'Pause')
        self.pause_button.on_clicked(self.toggle_pause)

        self.prev_button_ax = self.fig.add_axes([0.81, 0.25, 0.1, 0.075])
        self.prev_button = Button(self.prev_button_ax, 'Previous')
        self.prev_button.on_clicked(self.prev_frame)

        self.next_button_ax = self.fig.add_axes([0.81, 0.35, 0.1, 0.075])
        self.next_button = Button(self.next_button_ax, 'Next')
        self.next_button.on_clicked(self.next_frame)

        self.ax1.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.ani = FuncAnimation(self.fig, self.update_frame, interval=self.interval, blit=False)
        
        plt.show()

    def process_all_frames(self):
        if len(self.points) == 3:
            for i in range(self.num_frames):
                frame = self.image_array[i]

                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_image = frame

                mask = self.create_fan_mask(frame.shape)
                masked_image = np.bitwise_and(gray_image, mask)

                self.edited_image_array[i] = masked_image


    def create_fan_mask(self, shape):
        mask = np.zeros(shape, dtype=np.uint8)
        origin = self.points[0]
        start_point = self.points[1]
        end_point = self.points[2]

        axes = (int(np.linalg.norm(np.array(start_point) - np.array(origin))), 
                int(np.linalg.norm(np.array(end_point) - np.array(origin))))
        angle = 0
        start_angle = np.degrees(np.arctan2(start_point[1] - origin[1], start_point[0] - origin[0]))
        end_angle = np.degrees(np.arctan2(end_point[1] - origin[1], end_point[0] - origin[0]))

        cv2.ellipse(mask, origin, axes, angle, start_angle, end_angle, 255, -1)
        return mask

    def draw_fan_shape(self, frame):
        origin = self.points[0]
        start_point = self.points[1]
        end_point = self.points[2]

        if len(frame.shape) == 2:  # Grayscale image
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        axes = (int(np.linalg.norm(np.array(start_point) - np.array(origin))), 
                int(np.linalg.norm(np.array(end_point) - np.array(origin))))
        angle = 0
        start_angle = np.degrees(np.arctan2(start_point[1] - origin[1], start_point[0] - origin[0]))
        end_angle = np.degrees(np.arctan2(end_point[1] - origin[1], end_point[0] - origin[0]))

        cv2.ellipse(frame, origin, axes, angle, start_angle, end_angle, (0, 165, 255), 2)
        return frame

    def update_frame(self, *args):
        if not self.paused:
            self.current_frame = (self.current_frame + 1) % self.num_frames
        
        original_frame = self.image_array[self.current_frame]
        if len(self.points) == 3:
            original_frame_with_fan = self.draw_fan_shape(np.copy(original_frame))
        else:
            original_frame_with_fan = original_frame
        
        self.img1.set_data(original_frame_with_fan)
        self.img2.set_data(self.edited_image_array[self.current_frame])
        self.ax1.set_title(f'Original Frame {self.current_frame + 1}')
        self.ax2.set_title(f'Edited Frame {self.current_frame + 1}')
        self.fig.canvas.draw_idle()

    def save_image(self, event):
        edited_file_path = self.original_file_path.replace('.mha', '_edited.mha')
        new_image = sitk.GetImageFromArray(self.edited_image_array)
        for key in self.image.GetMetaDataKeys():
            new_image.SetMetaData(key, self.image.GetMetaData(key))
        sitk.WriteImage(new_image, edited_file_path)
        print(f'Saved edited image to {edited_file_path}')

    def toggle_pause(self, event):
        self.paused = not self.paused
        self.pause_button.label.set_text('Play' if self.paused else 'Pause')

    def prev_frame(self, event):
        self.paused = True
        self.current_frame = (self.current_frame - 1) % self.num_frames
        self.update_frame()

    def next_frame(self, event):
        self.paused = True
        self.current_frame = (self.current_frame + 1) % self.num_frames
        self.update_frame()

    def on_click(self, event):
        if event.inaxes == self.ax1:
            if len(self.points) < 3:
                self.points.append((int(event.xdata), int(event.ydata)))
                if len(self.points) == 3:
                    self.process_all_frames()
                    self.ax1.set_title(f'Original Frame {self.current_frame + 1}')
                    self.ax2.set_title(f'Edited Frame {self.current_frame + 1}')
                    self.img2.set_data(self.edited_image_array[self.current_frame])
                    self.update_frame()
            else:
                self.points = [(int(event.xdata), int(event.ydata))]  # Reset and start over if clicking after three points

                
file_path = r'seq_Dell_Vivid_Bharat_bilateral_falx_ventricles_D16_Bmode_config.mha'
MhaSlideshow(file_path, interval=50)  # Set interval to 500 milliseconds
'''
VolumeReconstructor.exe --config-file=C:\D\PlusB-bin\PlusLibData\ConfigFiles\CerebroSonic\scripts\seq_Dell_Vivid_Bharat_bilateral_falx_ventricles_D16_Bmode_config.xml --image-to-reference-transform=ImageToReference --source-seq-file=C:\D\PlusB-bin\PlusLibData\ConfigFiles\CerebroSonic\scripts\seq_Dell_Vivid_Bharat_bilateral_falx_ventricles_D16_Bmode_config_edited.mha --output-volume-file=C:\D\PlusB-bin\PlusLibData\ConfigFiles\CerebroSonic\scripts\vol_Dell_Vivid_Bharat_bilateral_falx_ventricles_D16_Bmode_config_edited.mha

'''
