import tkinter as tk
from tkinter import font
from env import (
    _MOVE_FWD_,
    _MOVE_BCK_,
    _MOVE_LEFT_,
    _MOVE_RIGHT_,
    _PAN_LEFT_,
    _PAN_RIGHT_,
    _GRIP_LEFT_,
    _GRIP_RIGHT_,
    _XTNFWD_LEFT_,
    _XTNFWD_RIGHT_,
    _PLLBCK_LEFT_,
    _PLLBCK_RIGHT_,
    _TILT_UP_,
    _TILT_DOWN_,
    _RSE_LEFT_,
    _LWR_LEFT_,
    _RSE_RIGHT_,
    _LWR_RIGHT_,
    _RESET_,
    _REQUEST_SCREENSHOT_,
    _REQUEST_ANNOTATION_,
    _REQUEST_JSON_,
)

from hand_reset import reset_hands_in_front2


def reset_hands():
    return reset_hands_in_front2(extra_elevation=-0.1, hand="left")


def on_key(event):
    controls = {
        'w': _MOVE_FWD_,
        's': _MOVE_BCK_,
        'a': _MOVE_LEFT_,
        'd': _MOVE_RIGHT_,
        'q': _PAN_LEFT_,
        'e': _PAN_RIGHT_,
        'u': _XTNFWD_LEFT_,
        'o': _XTNFWD_RIGHT_,
        'h': _PLLBCK_LEFT_,
        'k': _PLLBCK_RIGHT_,
        'b': _GRIP_LEFT_,
        'm': _GRIP_RIGHT_,
        'i': _TILT_UP_,
        'j': _TILT_DOWN_,
        'y': _RSE_LEFT_,
        'g': _LWR_LEFT_,
        'p': _RSE_RIGHT_,
        'l': _LWR_RIGHT_,
    }
    action = controls.get(event.keysym)
    if action:
        action()


root = tk.Tk()
root.title('Sari-Sari Environment V1 Agent Controls')
root.bind("<KeyPress>", on_key)

heading_font = font.Font(size=10, family='Courier', underline=True)
buttons_font = font.Font(size=7, family='Courier')

heading_frame = tk.Frame(root)
tk.Label(heading_frame, text='Agent Controls', bg='black', fg='white', font=heading_font).pack(side=tk.TOP)
heading_frame.pack()

controls_frame = tk.Frame(root)
tk.Button(controls_frame, text='Move\nForward\n(W)', bg='black', fg='white',
          font=buttons_font, command=_MOVE_FWD_, justify='left').grid(row=1, column=0, pady=2)

tk.Button(controls_frame, text='Move\nBackward\n(S)', bg='black', fg='white',
          font=buttons_font, command=_MOVE_BCK_, justify='left').grid(row=1, column=2, pady=2)

tk.Button(controls_frame, text='Move\nLeft\n(A)', bg='black', fg='white',
          font=buttons_font, command=_MOVE_LEFT_, justify='left').grid(row=1, column=1, pady=2)

tk.Button(controls_frame, text='Move\nRight\n(D)', bg='black', fg='white',
          font=buttons_font, command=_MOVE_RIGHT_, justify='left').grid(row=1, column=3, pady=2)

tk.Button(controls_frame, text='Pan\nLeft (Q)', bg='black', fg='white',
          font=buttons_font, command=_PAN_LEFT_, justify='left').grid(row=0, column=2, padx=2)

tk.Button(controls_frame, text='Pan\nRight (E)', bg='black', fg='white',
          font=buttons_font, command=_PAN_RIGHT_, justify='left').grid(row=0, column=3, padx=2)

tk.Button(controls_frame, text='Pan\nUp (I)', bg='black', fg='white',
          font=buttons_font, command=_TILT_UP_, justify='left').grid(row=0, column=0, padx=2)

tk.Button(controls_frame, text='Pan\nDown (J)', bg='black', fg='white',
          font=buttons_font, command=_TILT_DOWN_, justify='left').grid(row=0, column=1, padx=2)

controls_frame.pack(side=tk.TOP, pady=2)


hands_frame = tk.Frame(root)
tk.Button(hands_frame, text='Raise\nLeft Hand (Y)', bg='black', fg='white',
          font=buttons_font, command=_RSE_LEFT_, justify='left').grid(row=0, column=0, pady=2)

tk.Button(hands_frame, text='Raise\nRight Hand (P)', bg='black', fg='white',
          font=buttons_font, command=_RSE_RIGHT_, justify='left').grid(row=0, column=1, pady=2)

tk.Button(hands_frame, text='Extend\nLeft Hand (U)', bg='black', fg='white',
          font=buttons_font, command=_XTNFWD_LEFT_, justify='left').grid(row=1, column=0, pady=2)

tk.Button(hands_frame, text='Extend\nRight Hand (O)', bg='black', fg='white',
          font=buttons_font, command=_XTNFWD_RIGHT_, justify='left').grid(row=1, column=1, pady=2)

tk.Button(hands_frame, text='Retract\nLeft Hand (H)', bg='black', fg='white',
          font=buttons_font, command=_PLLBCK_LEFT_, justify='left').grid(row=2, column=0, padx=5)

tk.Button(hands_frame, text='Retract\nRight Hand (K)', bg='black', fg='white',
          font=buttons_font, command=_PLLBCK_RIGHT_, justify='left').grid(row=2, column=1, padx=5)

tk.Button(hands_frame, text='Toggle\nLeft Grip (B)', bg='black', fg='white',
          font=buttons_font, command=_GRIP_LEFT_, justify='left').grid(row=3, column=0, pady=2)

tk.Button(hands_frame, text='Toggle\nRight Grip (M)', bg='black', fg='white',
          font=buttons_font, command=_GRIP_RIGHT_, justify='left').grid(row=3, column=1, pady=2)

tk.Button(hands_frame, text='Lower\nLeft Hand (G)', bg='black', fg='white',
          font=buttons_font, command=_LWR_LEFT_, justify='left').grid(row=4, column=0, pady=2)

tk.Button(hands_frame, text='Lower\nRight Hand (L)', bg='black', fg='white',
          font=buttons_font, command=_LWR_RIGHT_, justify='left').grid(row=4, column=1, pady=2)

hands_frame.pack(side=tk.RIGHT, pady=3)

meta_frame = tk.Frame(root)
tk.Button(meta_frame, text='Reset\nEnvironment', bg='black', fg='white',
          font=buttons_font, command=_RESET_, justify='left').grid(row=0, column=0, pady=2)

tk.Button(meta_frame, text='Reset Hands\n(Not VR)', bg='black', fg='white',
          font=buttons_font, command=reset_hands, justify='left').grid(row=1, column=0, pady=2)

tk.Button(meta_frame, text='Request\nScreenshot', bg='black', fg='white',
          font=buttons_font, command=_REQUEST_SCREENSHOT_, justify='left').grid(row=2, column=0, pady=2)

tk.Button(meta_frame, text='Request\nAnnotation', bg='black', fg='white',
          font=buttons_font, command=_REQUEST_ANNOTATION_, justify='left').grid(row=3, column=0, pady=2)

tk.Button(meta_frame, text='Request JSON', bg='black', fg='white',
          font=buttons_font, command=_REQUEST_JSON_, justify='left').grid(row=4, column=0, pady=2)

meta_frame.pack(side=tk.LEFT, pady=3)

root.mainloop()
