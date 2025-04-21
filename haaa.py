import os
import google.generativeai as genai
import pytesseract
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import ImageGrab, Image, ImageTk, ImageSequence
import speech_recognition as sr
import pyttsx3
import threading
from ttkthemes import ThemedTk
from itertools import cycle
from openai import OpenAI
import mediapipe as mp

try:
    from sklearn.ensemble import RandomForestClassifier
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False

# Initialize APIs
try:
    genai.configure(api_key="")
    gemini_available = True
except:
    gemini_available = False

# Initialize OpenAI client
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# Initialize Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Initialize TTS Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

app_running = True
chat_history = ""
dark_mode = False
stop_event = threading.Event()

class GestureRecognizer:
    def __init__(self, root, response_text):
        self.root = root
        self.response_text = response_text
        self.cap = None
        self.running = False
        
        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.gesture_window = None
        self.video_label = None
        self.gesture_var = None
        self.last_gesture = None
        self.gesture_actions = {
            "Point 👆": lambda: self.send_gesture_command("Please explain this in detail."),
            "Victory ✌️": lambda: self.send_gesture_command("Can you compare these options?"),
            "Open 🖐️": lambda: self.clear_chat(),
            "Fist 👊": lambda: self.stop_gesture_operations()
        }

    def send_gesture_command(self, command):
        self.response_text.insert(tk.END, f"\nGesture command: {command}\n", "system")
        self.response_text.see(tk.END)
        user_text.insert(tk.END, command)
        send_message()

    def clear_chat(self):
        global chat_history
        chat_history = ""
        response_text.delete("1.0", tk.END)
        user_text.delete("1.0", tk.END)

    def stop_gesture_operations(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.gesture_window:
            self.gesture_window.destroy()

    def count_fingers(self, landmarks):
        finger_tips = [8, 12, 16, 20]
        finger_dips = [6, 10, 14, 18]
        count = 0
        for tip, dip in zip(finger_tips, finger_dips):
            if landmarks[tip].y < landmarks[dip].y:
                count += 1

        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        wrist = landmarks[0]

        if thumb_tip.x < thumb_ip.x:
            if thumb_tip.x < wrist.x:
                count += 1
        else:
            if thumb_tip.x > wrist.x:
                count += 1

        return count

    def detect_gesture(self, finger_count):
        gestures = {
            0: "Fist 👊",
            1: "Point 👆",
            2: "Victory ✌️",
            3: "Three 🤟",
            4: "Four 🖖",
            5: "Open 🖐️"
        }
        return gestures.get(finger_count, "🤖 Unknown")

    def show_gesture_window(self):
        if self.gesture_window and tk.Toplevel.winfo_exists(self.gesture_window):
            self.gesture_window.lift()
            return

        self.gesture_window = tk.Toplevel(self.root)
        self.gesture_window.title("🤖 Gesture Recognition System")
        self.gesture_window.geometry("900x700")
        self.gesture_window.resizable(False, False)
        self.gesture_window.protocol("WM_DELETE_WINDOW", self.stop_gesture_operations)
        self.gesture_window.configure(bg="#1c1c1c")

        # Video display
        self.video_label = tk.Label(self.gesture_window, bg="black", bd=3, relief="ridge")
        self.video_label.place(x=150, y=60, width=600, height=400)

        # Control buttons
        start_btn = tk.Button(
            self.gesture_window, text="🚀 Start", 
            font=('Helvetica', 12, 'bold'), bg="#00FFAB", fg="black",
            activebackground="#32e0c4", command=self.start_camera, 
            relief='raised', bd=2, width=15
        )
        start_btn.place(x=250, y=480)

        stop_btn = tk.Button(
            self.gesture_window, text="🛑 Stop", 
            font=('Helvetica', 12, 'bold'), bg="#FF6464", fg="white",
            activebackground="#ff4d6d", command=self.stop_gesture_operations, 
            state=tk.DISABLED, relief='raised', bd=2, width=15
        )
        stop_btn.place(x=550, y=480)

        # Gesture display
        self.gesture_var = tk.StringVar(value="Gesture: None")
        gesture_label = tk.Label(
            self.gesture_window, textvariable=self.gesture_var,
            font=('Helvetica', 22, 'bold'), fg="#00FFD1", bg="#1c1c1c"
        )
        gesture_label.place(x=330, y=530)

        # Status bar
        self.status_var = tk.StringVar(value="🔵 System Ready")
        status_label = tk.Label(
            self.gesture_window, textvariable=self.status_var,
            font=('Helvetica', 14, 'bold'), bg="#121212", fg="#00FFAB", 
            relief="sunken", anchor="w"
        )
        status_label.place(x=0, y=670, relwidth=1, height=30)

        # Gesture guide
        guide_text = """Gesture Guide:
        - Pointing (1 finger): Ask for explanation
        - Victory (2 fingers): Request comparison
        - Open Hand: Clear chat
        - Fist: Stop gesture recognition"""
        guide_label = tk.Label(
            self.gesture_window, text=guide_text,
            font=('Helvetica', 10), justify=tk.LEFT, bg="#1c1c1c", fg="white"
        )
        guide_label.place(x=300, y=580)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return
            
        self.running = True
        self.status_var.set("🟢 Camera Started")
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_gesture_operations()
            return
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                count = self.count_fingers(hand_landmarks.landmark)
                gesture = self.detect_gesture(count)
                self.gesture_var.set(f"Gesture: {gesture}")
                
                # Execute action if gesture changed
                if gesture != self.last_gesture and gesture in self.gesture_actions:
                    self.gesture_actions[gesture]()
                self.last_gesture = gesture
        else:
            self.gesture_var.set("Gesture: No Hand ✋")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)
        
        if self.running:
            self.gesture_window.after(15, self.update_frame)

class AnimatedButton(tk.Button):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.default_bg = self['bg']
        self.default_fg = self['fg']
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_enter(self, e):
        self.pulse_animation()
        
    def on_leave(self, e):
        self.stop_pulse()
        
    def pulse_animation(self, step=0):
        if step > 10:
            return
        factor = 0.9 + (0.1 * (1 + (step % 5 - 2.5)/5))
        r, g, b = self.winfo_rgb(self.default_bg)
        r = min(65535, int(r * factor))
        g = min(65535, int(g * factor))
        b = min(65535, int(b * factor))
        color = f"#{r:04x}{g:04x}{b:04x}"[0:7]
        self.config(bg=color)
        self.after(50, lambda: self.pulse_animation(step + 1))
        
    def stop_pulse(self):
        self.config(bg=self.default_bg, fg=self.default_fg)

def authenticate():
    # Create authentication window
    auth_window = tk.Toplevel(root)
    auth_window.title("Authentication Required")
    auth_window.geometry("400x300")
    auth_window.resizable(False, False)
    auth_window.grab_set()
    
    # Center the window
    window_width = auth_window.winfo_reqwidth()
    window_height = auth_window.winfo_reqheight()
    position_right = int(auth_window.winfo_screenwidth()/2 - window_width/2)
    position_down = int(auth_window.winfo_screenheight()/2 - window_height/2)
    auth_window.geometry(f"+{position_right}+{position_down}")
    
    # Style the authentication window
    auth_window.configure(bg="#f5f5f5")
    
    # Add logo/icon
    try:
        logo_img = Image.open("AS.png").resize((80, 80), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_img)
        logo_label = tk.Label(auth_window, image=logo_photo, bg="#f5f5f5")
        logo_label.image = logo_photo
        logo_label.pack(pady=(20, 10))
    except:
        logo_label = tk.Label(auth_window, text="🔒", font=("Arial", 40), bg="#f5f5f5")
        logo_label.pack(pady=(20, 10))
    
    # Add title
    title_label = tk.Label(auth_window, text="Secure Login", font=("Segoe UI", 16, "bold"), bg="#f5f5f5")
    title_label.pack()
    
    # Add subtitle
    subtitle_label = tk.Label(auth_window, text="Please enter your password to continue", 
                            font=("Segoe UI", 10), bg="#f5f5f5", fg="#666666")
    subtitle_label.pack(pady=(0, 20))
    
    # Add password frame
    password_frame = tk.Frame(auth_window, bg="#f5f5f5")
    password_frame.pack(pady=10)
    
    password_label = tk.Label(password_frame, text="Password:", font=("Segoe UI", 11), bg="#f5f5f5")
    password_label.pack(side=tk.LEFT, padx=(0, 10))
    
    password_entry = tk.Entry(password_frame, show="*", font=("Segoe UI", 11), 
                            width=20, bd=2, relief=tk.GROOVE, highlightthickness=0)
    password_entry.pack(side=tk.LEFT)
    password_entry.focus_set()
    
    # Add login button
    login_btn = tk.Button(auth_window, text="Login", font=("Segoe UI", 11, "bold"), 
                         bg="#4CAF50", fg="white", activebackground="#45a049",
                         command=lambda: verify_password(password_entry.get(), auth_window),
                         width=15, height=1, bd=0)
    login_btn.pack(pady=15, ipadx=10, ipady=5)
    
    # Add error label (initially empty)
    error_label = tk.Label(auth_window, text="", font=("Segoe UI", 9), bg="#f5f5f5", fg="#e74c3c")
    error_label.pack()
    
    # Bind Enter key to login
    auth_window.bind('<Return>', lambda event: verify_password(password_entry.get(), auth_window))
    
    def verify_password(password, window):
        if password == "sankalp":
            window.destroy()
            root.deiconify()
        else:
            error_label.config(text="Incorrect password! Please try again.")
            password_entry.delete(0, tk.END)
            # Shake animation for wrong password
            for _ in range(3):
                auth_window.geometry(f"+{position_right+5}+{position_down}")
                auth_window.update()
                auth_window.after(50)
                auth_window.geometry(f"+{position_right-5}+{position_down}")
                auth_window.update()
                auth_window.after(50)
            auth_window.geometry(f"+{position_right}+{position_down}")
    
    # Hide main window until authenticated
    root.withdraw()
    auth_window.protocol("WM_DELETE_WINDOW", lambda: (root.destroy(), auth_window.destroy()))

def toggle_theme():
    global dark_mode
    dark_mode = not dark_mode
    if dark_mode:
        root.set_theme("equilux")
        response_text.config(bg="#2C3E50", fg="#ECF0F1")
        user_text.config(bg="#34495E", fg="#ECF0F1")
        button_frame.config(bg="#2C3E50")
        frame.config(bg="#34495E")
        for btn in all_buttons:
            btn.config(fg="#ECF0F1")
    else:
        root.set_theme("breeze")
        response_text.config(bg="#ECF0F1", fg="#2C3E50")
        user_text.config(bg="#FFFFFF", fg="#2C3E50")
        button_frame.config(bg="#ECF0F1")
        frame.config(bg="#BDC3C7")
        for btn in all_buttons:
            btn.config(fg="#2C3E50")

def animated_text_display(widget, text, tag):
    if stop_event.is_set():
        return
    widget.insert("end", "\n", tag)
    for char in text:
        if stop_event.is_set():
            break
        widget.insert("end", char, tag)
        widget.update_idletasks()
        widget.after(50)
    widget.insert("end", "\n")

def speak_text(text):
    if not stop_event.is_set():
        engine.say(text)
        engine.runAndWait()

def handle_response(response):
    threading.Thread(target=lambda: animated_text_display(response_text, f"Assistant: {response}\n", "bot")).start()
    threading.Thread(target=speak_text, args=(response,)).start()

def is_comparison_query(prompt):
    comparison_phrases = [
        "difference between",
        "compare",
        "vs",
        "versus",
        "similarities and differences",
        "pros and cons"
    ]
    return any(phrase in prompt.lower() for phrase in comparison_phrases)

def get_ai_response(prompt):
    global chat_history
    try:
        if stop_event.is_set():
            return "[Operation Stopped]"
            
        if is_comparison_query(prompt):
            system_message = """When asked about comparisons, respond with a well-formatted markdown table.
            Use clear columns like 'Feature', 'Option A', 'Option B' etc.
            Include all relevant comparison points in a structured manner."""
            
            if gemini_available:
                model = genai.GenerativeModel("gemini-1.5-pro")
                response = model.generate_content(
                    f"{system_message}\n\nUser query: {prompt}",
                    stream=True
                )
                response.resolve()
                response_text = response.text
            else:
                completion = openai_client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "Comparison Table Generator",
                    },
                    model="deepseek/deepseek-r1:free",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = completion.choices[0].message.content
            
            if "|" in response_text:
                response_text = "\nComparison Results:\n" + response_text
        else:
            if gemini_available:
                model = genai.GenerativeModel("gemini-1.5-pro")
                chat_history += f"You: {prompt}\n"
                response = model.generate_content(chat_history + "\nAI:")
                response_text = response.text if response.text else "Sorry, I didn't understand."
            else:
                completion = openai_client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "AI Chatbot",
                    },
                    model="deepseek/deepseek-r1:free",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                response_text = completion.choices[0].message.content
        
        chat_history += f"AI: {response_text}\n"
        return response_text
    except Exception as e:
        return f"Error: {str(e)}"

def send_message(event=None):
    stop_event.clear()
    user_input = user_text.get("1.0", "end-1c").strip()
    if user_input:
        animated_text_display(response_text, f"You: {user_input}\n", "user")
        user_text.delete("1.0", tk.END)
        threading.Thread(target=lambda: handle_response(get_ai_response(user_input))).start()

def stop_operations():
    stop_event.set()
    engine.stop()
    response_text.insert(tk.END, "\n[Stopped] Ready for next query...\n", "system")
    response_text.see(tk.END)

def scan_image():
    stop_event.clear()
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        try:
            img = cv2.imread(file_path)
            text = pytesseract.image_to_string(img)
            if text.strip():
                user_text.insert(tk.END, text)
                send_message()
            else:
                messagebox.showerror("Error", "No text detected in the image.")
        except Exception as e:
            messagebox.showerror("Error", f"Image processing failed: {e}")

def take_screenshot():
    stop_event.clear()
    screenshot = ImageGrab.grab()
    screenshot.save("screenshot.png")
    messagebox.showinfo("Screenshot", "Screenshot saved as 'screenshot.png'.")

def clear_chat():
    global chat_history
    chat_history = ""
    response_text.delete("1.0", tk.END)
    user_text.delete("1.0", tk.END)

def stop_app():
    global app_running
    app_running = False
    engine.stop()
    root.quit()

def voice_input():
    stop_event.clear()
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        messagebox.showinfo("Voice Input", "Speak now...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            user_text.insert(tk.END, text)
            send_message()
        except sr.UnknownValueError:
            messagebox.showerror("Error", "Could not understand the audio.")
        except sr.RequestError:
            messagebox.showerror("Error", "Error connecting to voice recognition service.")

def start_gesture_recognition():
    gesture_recognizer = GestureRecognizer(root, response_text)
    gesture_recognizer.show_gesture_window()

# Enhanced GUI Setup
root = ThemedTk(theme="breeze")
root.title("AI Chatbot - Enhanced Interface")
root.geometry("900x750")
root.configure(bg="#ECF0F1")

# Custom title bar
title_frame = tk.Frame(root, bg="#3498DB", height=50)
title_frame.pack(fill=tk.X)

title_label = tk.Label(title_frame, text="sankalp Assistant", font=("Helvetica", 16, "bold"), 
                      bg="#3498DB", fg="white")
title_label.pack(side=tk.LEFT, padx=20)

# Main content frame
main_frame = tk.Frame(root, bg="#ECF0F1")
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Chat display area with rounded corners
frame = tk.Frame(main_frame, bg="#BDC3C7", bd=2, relief=tk.GROOVE)
frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

scrollbar = tk.Scrollbar(frame)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

response_text = tk.Text(frame, bg="#ECF0F1", fg="#2C3E50", font=("Segoe UI", 12), 
                       wrap="word", height=15, yscrollcommand=scrollbar.set,
                       padx=15, pady=15, relief=tk.FLAT)
response_text.tag_config("user", foreground="#2980B9", font=("Segoe UI", 12, "bold"))
response_text.tag_config("bot", foreground="#27AE60", font=("Segoe UI", 12))
response_text.tag_config("system", foreground="#E74C3C", font=("Segoe UI", 10, "italic"))
response_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.config(command=response_text.yview)

# User input area with rounded corners
input_frame = tk.Frame(main_frame, bg="#ECF0F1")
input_frame.pack(fill=tk.X, pady=(0, 10))

user_text = tk.Text(input_frame, height=4, font=("Segoe UI", 12), bg="#FFFFFF", 
                   fg="#2C3E50", wrap=tk.WORD, relief=tk.GROOVE, 
                   padx=10, pady=10)
user_text.pack(fill=tk.X)
user_text.bind("<Return>", send_message)

# Button panel with modern styling
button_frame = tk.Frame(main_frame, bg="#ECF0F1")
button_frame.pack(fill=tk.X)

# Button styles
button_styles = {
    "Send": {"bg": "#2ECC71", "activebg": "#27AE60"},
    "Scan Image": {"bg": "#3498DB", "activebg": "#2980B9"},
    "Screenshot": {"bg": "#9B59B6", "activebg": "#8E44AD"},
    "Voice Input": {"bg": "#F39C12", "activebg": "#E67E22"},
    "Clear": {"bg": "#E74C3C", "activebg": "#C0392B"},
    "Stop": {"bg": "#E67E22", "activebg": "#D35400"},
    "Toggle Theme": {"bg": "#34495E", "activebg": "#2C3E50"},
    "Gestures": {"bg": "#1ABC9C", "activebg": "#16A085"}
}

all_buttons = []

# Create buttons with consistent styling
for text, command in [
    ("Send", send_message),
    ("Scan Image", scan_image),
    ("Screenshot", take_screenshot),
    ("Voice Input", voice_input),
    ("Clear", clear_chat),
    ("Stop", stop_operations),
    ("Toggle Theme", toggle_theme),
    ("Gestures", start_gesture_recognition)
]:
    btn = AnimatedButton(
        button_frame,
        text=text,
        command=command,
        bg=button_styles[text]["bg"],
        fg="white",
        activebackground=button_styles[text]["activebg"],
        activeforeground="white",
        relief=tk.FLAT,
        borderwidth=0,
        width=15,
        font=("Segoe UI", 10, "bold"),
        padx=10,
        pady=5
    )
    btn.pack(side=tk.LEFT, padx=5, pady=5)
    all_buttons.append(btn)

# Status bar
status_bar = tk.Label(root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W,
                     font=("Segoe UI", 9), bg="#BDC3C7", fg="#2C3E50")
status_bar.pack(fill=tk.X)

authenticate()

root.mainloop()
