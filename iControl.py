import tkinter as tk
from tkinter import ttk
import threading
import time
from time import sleep
from joblib import load
import warnings
warnings.filterwarnings('ignore')

import serial
import pandas as pd
import numpy as np
from pynput.keyboard import Key, Controller as keyboard_controller
import statistics
import random

from PIL import Image, ImageTk, ImageEnhance



keyboard = keyboard_controller()

# Initialize the GUI application
root = tk.Tk()
root.title("iControl")
root.geometry("800x600")
root.configure(bg="#4C257B")

# Load and initialize the background
background_image = Image.open("GUI\\background.png").resize((800, 600), Image.Resampling.LANCZOS)
background_photo = ImageTk.PhotoImage(background_image)
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=background_photo, anchor="nw")

# Load and initialize the CD image
original_cd_image = Image.open("GUI\\cd.png").convert("RGBA")
resized_cd_image = original_cd_image.resize((200, 200), Image.Resampling.LANCZOS)
cd_image = ImageTk.PhotoImage(resized_cd_image)
cd_label = tk.Label(canvas, image=cd_image, bd=0, highlightthickness=0, bg="#4C257B")

cd_spinning = True
cd_angle = 0

def rotate_cd():
    global cd_angle, cd_image, cd_spinning
    if cd_spinning:
        cd_angle = (cd_angle - 5) % 360
        rotated_image = resized_cd_image.rotate(cd_angle)
        cd_image = ImageTk.PhotoImage(rotated_image)
        cd_label.configure(image=cd_image)
    root.after(50, rotate_cd)

def start_cd_spin():
    global cd_spinning
    cd_spinning = True

def stop_cd_spin():
    global cd_spinning
    cd_spinning = False

def change_cd_color():
    global cd_image, resized_cd_image
    enhancer = ImageEnhance.Color(resized_cd_image)
    factor = random.uniform(-100, 100)  # Change this range for different color intensities
    resized_cd_image = enhancer.enhance(factor)
    cd_image = ImageTk.PhotoImage(resized_cd_image)
    cd_label.configure(image=cd_image)

# Start the CD spinning
rotate_cd()

# Function to handle the loop started by the "Start" button
def read_arduino(ser, inputBufferSize):
    data = ser.read(inputBufferSize)
    out = [(int(data[i])) for i in range(0, len(data))]
    return out

def process_data(data):
    data_in = np.array(data)
    result = []
    i = 1
    while i < len(data_in) - 1:
        if data_in[i] > 127:
            # Found beginning of frame - Extract one sample from 2 bytes
            intout = (np.bitwise_and(data_in[i], 127)) * 128
            i = i + 1
            intout = intout + data_in[i]
            result = np.append(result, intout)
        i = i + 1
    return result

def process_gaussian_fft(t, data_t, sigma_gauss):
    nfft = len(data_t)  # number of points
    dt = t[1] - t[0]  # time interval
    maxf = 1 / dt  # maximum frequency
    df = 1 / np.max(t)  # frequency interval
    f_fft = np.arange(-maxf / 2, maxf / 2 + df, df)  # define frequency domain

    # Do FFT
    data_f = np.fft.fftshift(np.fft.fft(data_t))  # FFT of data

    # Gaussian filter
    gauss_filter = np.exp(-(f_fft) ** 2 / sigma_gauss ** 2)  # gaussian filter used
    data_f_filtered = data_f * gauss_filter  # gaussian filter spectrum in frequency domain
    data_t_filtered = np.fft.ifft(np.fft.ifftshift(data_f_filtered))  # bring filtered signal in time domain
    return data_t_filtered

def stdev(scaled_list):
    return statistics.stdev(scaled_list[:4800])

def predict_sd(scaled_list):
    if scaled_list.iloc[-1] >= 35:
        prediction = 1
    else:
        prediction = 0
    return prediction

def prepare_for_tsfresh(df):
    final_df = df.transpose()
    final_df = final_df.groupby(np.arange(len(final_df)) // 25).first()
    final_df = final_df.transpose ()
    final_df = final_df.melt(ignore_index=False).reset_index()
    final_df.columns = ['unique_id', 'ds', 'y']
    final_df["ds"] = final_df["ds"].astype(int)
    final_df = final_df.set_index('ds')
    return final_df

def arduino():
    global monitoring
    model = load('models\\live_rf_b_l_r_ne_combined.joblib')
    baudrate = 230400
    cport = 'COM8'  # set the correct port before you run it
    ser = serial.Serial(port=cport, baudrate=baudrate)
    inputBufferSize = 10000
    data_window = []
    max_loops = 500
    activated = max_loops
    pp = 1

    started = False
    lastblink = False

    while monitoring:
        
        #Reading data from ardiuno
        data = read_arduino(ser,inputBufferSize)
        
        if activated%100 == 0 and activated > max_loops: #roughly every second, remind the user that it's off
            print("iControl is OFF - Blink Twice to Enable iControl")
        
        #testing a data length threshold
        if len(data) > 10:
            if started == False:
                started = True
                print("Ready to go!")
            data_temp = process_data(data)        
            data_window.insert(0, data_temp) #Creating increments
            
            if len(data_window) == 1: #First window
                window = data_window[0]
            
            else: #Join the windows 
                window = np.concatenate((data_window[1], data_window[0]))
                del data_window[-1]

            T = inputBufferSize*2/20000.0*np.linspace(0,1,len(window)) # define temporary time array OF THE WINDOW
            sigma_gauss = 25
            
            data_temp_filtered = process_gaussian_fft(T,window,sigma_gauss) #filter data
            
            #Convert data in a pd df
            real_parts = [num.real for num in data_temp_filtered]
            df_filt = pd.DataFrame([real_parts])

            #Processing here to cut off the spikes
            if len(df_filt.columns)>5000: #check that we have a correctly sized window
                #Cut off spike by deleting first and last 200 rows

                df_filt = df_filt.iloc[:, 196: 9991-196]
                df_filt.columns = pd.RangeIndex(df_filt.columns.size)
                df_filt.columns = ['col_' + str(col) for col in df_filt.columns]

                # Making a duplicate dataframe to use for the specific event classifier later
                df2 = df_filt.copy()
                df_filt['sd'] = df_filt.apply(stdev, axis = 1) #calculating the sd
                threshold = 35 #Manually set from training 
                prediction = df_filt.apply(predict_sd, axis = 1).iloc[0]

                if prediction == 1: #If event                
                    event = model.predict(df2)

                    #Deleting the next window to stop doubling up
                    del data_window[-1]

                    #Actual output
                    if event == 0:
                        if activated < max_loops: #if double blink has been recorded in last 50 loop iterations
                            print("Blink Detected - Play/Pausing")
                            activated = 0 #reset the loop counter
                            keyboard.press(Key.media_play_pause)
                            if pp % 2 == 0:
                                show_pause_button()
                                stop_cd_spin()
                            else:
                                show_play_button()
                                start_cd_spin()
                            pp += 1
                        else:
                            if lastblink == True: #if the previous action was also a blink
                                activated = 0
                                print("Double Blink Detected - iControl ON")
                                action_symbol_label.configure(text="Scanning")
                                lastblink = False
                            else:
                                lastblink = True
                            print("Blink Detected - Blink Twice to Enable iControl")
                        sleep(1) #adding a half a second delay after a blink to prevent doubling up
                        activated += 50 #to make up for the half second delay
                    elif event == 1:
                        lastblink = False
                        if pp % 2 == 1:
                            start_cd_spin()
                            pp += 1
                        if activated < max_loops:
                            print("Left Detected - Skipping to Previous Track")
                            activated = 0
                            keyboard.press(Key.media_previous)
                            sleep(0.5)
                            keyboard.press(Key.media_previous)
                            show_back_button()
                            change_cd_color()
                            start_cd_spin()
                        else:
                            print("Left Detected - Blink Twice to Enable iControl")
                        sleep(1)
                    elif event == 2:
                        lastblink = False
                        if pp % 2 == 1:
                            start_cd_spin()
                            pp += 1
                        if activated < max_loops:
                            print("Right Detected - Skipping to Next Track")
                            activated = 0
                            keyboard.press(Key.media_next)
                            show_skip_button()
                            change_cd_color()
                            start_cd_spin()
                        else:
                            print("Right Detected - Blink Twice to Enable iControl")
                        sleep(1)
                    if activated == max_loops:
                        action_symbol_label.configure(text="       ")
                activated += 1

#Gui functions
def start_monitoring():
    global monitoring
    if not monitoring:
        monitoring = True
        thread = threading.Thread(target=arduino, daemon=True)
        thread.start()

def stop_monitoring():
    global monitoring
    monitoring = False

def show_skip_button():
    action_symbol_label.configure(text="⏭")

def show_back_button():
    action_symbol_label.configure(text="⏮")

def show_play_button():
    action_symbol_label.configure(text=" ▶ ")

def show_pause_button():
    action_symbol_label.configure(text="⏸")

# Font styles
symbol_font = ("Arial", 60, "bold")
text_font = ("Arial", 16, "bold")

# Dynamic action labels
action_text_label = tk.Label(canvas, text="Last action:", font=text_font, bg="#D66E3A", fg="white")
action_text_label.place(relx=0.5, rely=0.05, anchor='n')

action_symbol_label = tk.Label(canvas, text="       ", font=symbol_font, bg="#D66E3A", fg="white")
action_symbol_label.place(relx=0.5, rely=0.12, anchor='n')

# CD label in the middle, centered
cd_label.place(relx=0.5, rely=0.5, anchor='center')

# Button styles
button_style_control = {
    "font": ("Arial", 16, "bold"),
    "width": 10,
    "height": 3,
    "foreground": "white",
    "background": "#D66E3A"  # Orangey-red color
}

# Start and Stop buttons at the bottom
start_button = tk.Button(canvas, text="START", command=start_monitoring, **button_style_control)
start_button.place(relx=0.3, rely=0.9, anchor='s')

stop_button = tk.Button(canvas, text="STOP", command=stop_monitoring, **button_style_control)
stop_button.place(relx=0.7, rely=0.9, anchor='s')

monitoring = False
root.mainloop()

