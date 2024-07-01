import tkinter as tk
from PIL import Image, ImageTk
import requests


# Flask server URL (update with your actual server address)
FLASK_SERVER_URL = 'http://127.0.0.1:5000'


def send_request(action):
    try:
        response = requests.post(f'{FLASK_SERVER_URL}/toggle', json={'action': action})
        if response.status_code == 200:
            print(f"Action '{action}' sent successfully!")
        else:
            print(f"Failed to send action '{action}'. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending request: {e}")

# Create the main window
root = tk.Tk()
root.title("WAF GUI")
# Set the window size
root.geometry("500x500")
# Load and resize the logo image
try:
    logo_image = Image.open("/Users/zeyadhassan/Desktop/WAF/waf_pic.png")
    logo_image = logo_image.resize((300, 300), Image.LANCZOS)
    logo_photo = ImageTk.PhotoImage(logo_image)
except Exception as e:
    print(f"Error loading logo: {e}")
    logo_photo = None

# Function for ON button click
def on_button_click():
    print("ON button clicked")
    send_request('ON')

# Function for OFF button click
def off_button_click():
    print("OFF button clicked")
    send_request('OFF')

# Create a label for the logo image
logo_label = tk.Label(root, image=logo_photo)
logo_label.pack(pady=20)

# Create ON and OFF buttons
on_button = tk.Button(root, text="ON", command=on_button_click)
off_button = tk.Button(root, text="OFF", command=off_button_click)

# Pack the buttons with some spacing
on_button.pack(side=tk.LEFT, padx=90)
off_button.pack(side=tk.LEFT, padx=100)

# Run the main loop
root.mainloop()
