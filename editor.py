import customtkinter as ctk
from tkinter import filedialog, Canvas, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageFont, ImageDraw, ImageFilter
import easyocr
import threading
import sys
import os
import glob

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class RealTimeEditor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Local God-Mode Editor (Smart Fonts)")
        self.geometry("1400x950")
        
        # --- CORE STATE ---
        self.cv_image = None         
        self.display_image = None    
        self.clean_image = None      
        self.boxes = []
        self.history = []            
        self.reader = None
        
        # --- EDITING STATE ---
        self.active_box_idx = -1
        self.is_live_editing = False 
        self.picked_color = (0, 0, 0)
        self.is_picking_color = False
        
        # Font System
        self.system_fonts = self.find_common_fonts()
        self.current_font_path = self.system_fonts.get("Arial (Default)", None)
        
        self.setup_sidebar()
        self.setup_canvas()
        
    def find_common_fonts(self):
        """Auto-detects common bill/medical fonts on Windows"""
        font_dict = {}
        # Common Windows Font Paths
        win_path = r"C:\Windows\Fonts"
        
        # The "Big 5" Mapping
        targets = {
            "Arial": "arial.ttf",
            "Times New Roman (Medical)": "times.ttf",
            "Courier New (Bills)": "cour.ttf",
            "Segoe UI (Windows Apps)": "segoeui.ttf",
            "Verdana": "verdana.ttf",
            "Calibri": "calibri.ttf",
            "Consolas (Code)": "consola.ttf",
            "Tahoma": "tahoma.ttf"
        }
        
        for name, filename in targets.items():
            full_path = os.path.join(win_path, filename)
            if os.path.exists(full_path):
                font_dict[name] = full_path
            else:
                # Try finding it roughly
                candidates = glob.glob(os.path.join(win_path, filename[:3] + "*.ttf"))
                if candidates:
                    font_dict[name] = candidates[0]
                    
        # Add a default fallback
        font_dict["Default"] = None
        return font_dict

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="SMART EDITOR", font=("Arial", 20, "bold")).pack(pady=15)
        
        # File Operations
        frame_io = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame_io.pack(fill="x", padx=10)
        ctk.CTkButton(frame_io, text="Open Image", command=self.load_image, width=130).pack(side="left", padx=5)
        ctk.CTkButton(frame_io, text="Save Result", command=self.save_image, fg_color="green", width=130).pack(side="right", padx=5)
        ctk.CTkButton(self.sidebar, text="Undo", command=self.undo, fg_color="gray").pack(pady=5, padx=15, fill="x")

        # Analysis
        ctk.CTkLabel(self.sidebar, text="1. PREPARE", font=("Arial", 12, "bold"), text_color="gray").pack(pady=(15, 5))
        self.btn_detect = ctk.CTkButton(self.sidebar, text="Detect Text", command=self.start_detection)
        self.btn_detect.pack(pady=2, padx=15, fill="x")
        self.status_label = ctk.CTkLabel(self.sidebar, text="Ready", text_color="gray", font=("Arial", 10))
        self.status_label.pack(pady=2)

        # Smart Assets
        ctk.CTkLabel(self.sidebar, text="2. SMART MATCHING", font=("Arial", 12, "bold"), text_color="gray").pack(pady=(15, 5))
        
        # Font Dropdown
        ctk.CTkLabel(self.sidebar, text="Select Font Style:", font=("Arial", 10), anchor="w").pack(padx=20, fill="x")
        self.font_var = ctk.StringVar(value="Arial")
        self.font_dropdown = ctk.CTkComboBox(self.sidebar, values=list(self.system_fonts.keys()), command=self.change_font)
        self.font_dropdown.pack(pady=2, padx=15, fill="x")
        
        # Load Custom Button (Backup)
        ctk.CTkButton(self.sidebar, text="📂 Load Custom .TTF", command=self.load_custom_font, fg_color="#444", height=20, font=("Arial", 10)).pack(pady=2, padx=15, fill="x")

        self.btn_pick = ctk.CTkButton(self.sidebar, text="🖌️ Pick Color", command=self.toggle_color_picker, fg_color="#444", height=25)
        self.btn_pick.pack(pady=5, padx=15, fill="x")
        self.color_preview = ctk.CTkLabel(self.sidebar, text="Color: RGB(0,0,0)", fg_color="#000000", height=20, corner_radius=5)
        self.color_preview.pack(pady=2, padx=15, fill="x")

        # Live Controls
        ctk.CTkLabel(self.sidebar, text="3. LIVE TWEAKING", font=("Arial", 12, "bold"), text_color="#3498db").pack(pady=(15, 5))
        
        self.entry_text = ctk.CTkEntry(self.sidebar, placeholder_text="Select a box first...")
        self.entry_text.pack(pady=5, padx=15, fill="x")
        self.entry_text.bind("<KeyRelease>", self.on_text_change)

        self.add_slider("Font Size", 8, 200, 20, "font_size")
        self.add_slider("Spacing", -10, 20, 0, "spacing")
        self.add_slider("Y-Offset", -50, 50, 0, "offset_y")
        self.add_slider("X-Offset", -50, 50, 0, "offset_x")
        self.add_slider("Opacity", 0, 255, 255, "opacity")
        self.add_slider("Blur", 0, 4, 0, "blur")
        
        self.btn_commit = ctk.CTkButton(self.sidebar, text="✅ COMMIT CHANGES", command=self.commit_edit, fg_color="#3498db", state="disabled")
        self.btn_commit.pack(pady=20, padx=15, fill="x")

    def add_slider(self, label, min_val, max_val, default, attr_name):
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=0)
        header = ctk.CTkFrame(frame, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(header, text=label, font=("Arial", 10)).pack(side="left")
        val_lbl = ctk.CTkLabel(header, text=str(default), font=("Arial", 10, "bold"), text_color="#aaa")
        val_lbl.pack(side="right")

        def update_val(v):
            setattr(self, f"val_{attr_name}", v)
            fmt = f"{v:.1f}" if attr_name in ['blur','spacing'] else f"{int(v)}"
            val_lbl.configure(text=fmt)
            if self.is_live_editing:
                self.update_preview()

        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val, number_of_steps=(max_val-min_val)*2, command=update_val)
        slider.set(default)
        slider.pack(fill="x", pady=(0,5))
        setattr(self, f"val_{attr_name}", default)

    # --- FONT LOGIC ---
    def change_font(self, choice):
        self.current_font_path = self.system_fonts[choice]
        if self.is_live_editing:
            self.update_preview()

    def load_custom_font(self):
        path = filedialog.askopenfilename(filetypes=[("Font", "*.ttf;*.otf")])
        if path:
            name = os.path.basename(path)
            self.system_fonts[name] = path
            self.font_dropdown.configure(values=list(self.system_fonts.keys()))
            self.font_dropdown.set(name)
            self.change_font(name)

    # --- STANDARD LOGIC ---
    def setup_canvas(self):
        self.canvas_frame = ctk.CTkFrame(self, fg_color="#2b2b2b")
        self.canvas_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.canvas = Canvas(self.canvas_frame, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_click)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        self.cv_image = cv2.imread(path)
        if self.cv_image is None: return
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.display_image = Image.fromarray(self.cv_image)
        self.clean_image = None
        self.boxes = []
        self.is_live_editing = False
        self.show_image()

    def start_detection(self):
        self.status_label.configure(text="Scanning...")
        threading.Thread(target=self.run_ocr).start()

    def run_ocr(self):
        if self.reader is None: self.reader = easyocr.Reader(['en'], gpu=False)
        img_np = np.array(self.display_image)
        results = self.reader.readtext(img_np)
        self.boxes = []
        for (bbox, text, _) in results:
            (tl, tr, br, bl) = bbox
            self.boxes.append({'x': int(tl[0]), 'y': int(tl[1]), 'w': int(br[0]-tl[0]), 'h': int(br[1]-tl[1]), 'text': text})
        self.status_label.configure(text=f"Found {len(self.boxes)} items")
        self.after(0, self.show_image)

    def on_click(self, event):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.display_image.size
        scale = min(cw/iw, ch/ih)
        ox, oy = (cw - iw*scale)//2, (ch - ih*scale)//2
        mx, my = (event.x - ox) / scale, (event.y - oy) / scale

        if self.is_picking_color:
            if 0 <= mx < iw and 0 <= my < ih:
                self.picked_color = self.display_image.getpixel((int(mx), int(my)))
                hex_c = '#{:02x}{:02x}{:02x}'.format(*self.picked_color)
                self.color_preview.configure(fg_color=hex_c, text=f"{hex_c}")
                self.toggle_color_picker()
                if self.is_live_editing: self.update_preview()
            return

        if self.is_live_editing: self.commit_edit()

        for i, box in enumerate(self.boxes):
            if box['x'] < mx < box['x']+box['w'] and box['y'] < my < box['y']+box['h']:
                self.start_live_edit(i)
                return

    def start_live_edit(self, index):
        self.active_box_idx = index
        self.is_live_editing = True
        self.btn_commit.configure(state="normal", text="✅ COMMIT CHANGES", fg_color="#3498db")
        box = self.boxes[index]
        self.entry_text.delete(0, "end")
        self.entry_text.insert(0, box['text'])
        
        img_np = np.array(self.display_image)
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        pad = 4
        mask[max(0, box['y']-pad):box['y']+box['h']+pad, max(0, box['x']-pad):box['x']+box['w']+pad] = 255
        self.clean_image = Image.fromarray(cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA))
        self.val_font_size = int(box['h'] * 0.8)
        self.update_preview()

    def on_text_change(self, event):
        if self.is_live_editing: self.update_preview()

    def update_preview(self):
        if not self.clean_image or self.active_box_idx == -1: return
        base = self.clean_image.copy()
        txt_layer = Image.new('RGBA', base.size, (0,0,0,0))
        draw = ImageDraw.Draw(txt_layer)
        box = self.boxes[self.active_box_idx]
        text = self.entry_text.get()
        
        try:
            if self.current_font_path:
                font = ImageFont.truetype(self.current_font_path, int(self.val_font_size))
            else:
                font = ImageFont.truetype("arial.ttf", int(self.val_font_size))
        except:
            font = ImageFont.load_default()

        fill = (self.picked_color[0], self.picked_color[1], self.picked_color[2], int(self.val_opacity))
        x, y = box['x'] + self.val_offset_x, box['y'] + self.val_offset_y
        
        for char in text:
            draw.text((x, y), char, font=font, fill=fill)
            try: w = draw.textlength(char, font=font)
            except: w, _ = draw.textsize(char, font=font)
            x += w + self.val_spacing

        if self.val_blur > 0:
            txt_layer = txt_layer.filter(ImageFilter.GaussianBlur(radius=self.val_blur/2))
        base.paste(txt_layer, (0,0), txt_layer)
        self.display_image = base
        self.show_image()

    def commit_edit(self):
        if not self.is_live_editing: return
        self.history.append(self.display_image.copy())
        self.is_live_editing = False
        self.clean_image = None
        self.active_box_idx = -1
        self.btn_commit.configure(state="disabled", text="Select a box...", fg_color="gray")
        self.show_image()

    def undo(self):
        if self.history:
            self.display_image = self.history.pop()
            self.is_live_editing = False
            self.show_image()
            
    def save_image(self):
        if self.display_image is None: return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if path:
            self.display_image.save(path)
            messagebox.showinfo("Saved", "Image saved successfully!")

    def show_image(self):
        if self.display_image is None: return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600
        iw, ih = self.display_image.size
        scale = min(cw/iw, ch/ih)
        nw, nh = int(iw*scale), int(ih*scale)
        img = self.display_image.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        ox, oy = (cw-nw)//2, (ch-nh)//2
        self.canvas.create_image(ox + nw//2, oy + nh//2, image=self.tk_image)
        
        if not self.is_live_editing:
            for i, box in enumerate(self.boxes):
                cx, cy = box['x']*scale + ox, box['y']*scale + oy
                cw, ch = box['w']*scale, box['h']*scale
                self.canvas.create_rectangle(cx, cy, cx+cw, cy+ch, outline="#00ff00")
        elif self.active_box_idx != -1:
            box = self.boxes[self.active_box_idx]
            cx, cy = box['x']*scale + ox, box['y']*scale + oy
            cw, ch = box['w']*scale, box['h']*scale
            self.canvas.create_rectangle(cx, cy, cx+cw, cy+ch, outline="#3498db", width=2)

    def toggle_color_picker(self):
        self.is_picking_color = not self.is_picking_color
        self.btn_pick.configure(fg_color="#F39C12" if self.is_picking_color else "#444")
        self.canvas.configure(cursor="crosshair" if self.is_picking_color else "")

if __name__ == "__main__":
    app = RealTimeEditor()
    app.mainloop()