import customtkinter as ctk
from tkinter import filedialog, Canvas, messagebox, colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageFont, ImageDraw, ImageFilter, ImageEnhance
import easyocr
import threading
import os
import glob
import time

# --- MODERN THEME CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

COLORS = {
    'primary': '#6366f1', 'primary_hover': '#4f46e5', 'secondary': '#8b5cf6',
    'success': '#10b981', 'warning': '#f59e0b', 'danger': '#ef4444',
    'bg_dark': '#0f172a', 'bg_medium': '#1e293b', 'bg_light': '#334155',
    'text_primary': '#f1f5f9', 'text_secondary': '#94a3b8', 'border': '#475569'
}

class RealTimeEditor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Text Editor Pro - Optimized Real-Time")
        self.geometry("1700x950")
        self.configure(fg_color=COLORS['bg_dark'])
        
        # --- CORE STATE ---
        self.cv_image = None
        self.display_image = None
        self.original_display = None
        self.clean_image = None
        self.boxes = []
        self.history = []
        self.redo_stack = []
        self.reader = None
        self.showing_original = False
        
        # --- EDITING STATE ---
        self.active_box_idx = -1
        self.is_live_editing = False
        self.picked_color = (0, 0, 0)
        self.is_picking_color = False
        self.original_color = (0, 0, 0)
        self.temp_edits = {}
        self.background_noise = None
        
        # --- OPTIMIZATION ---
        self.update_pending = False
        self.last_update_time = 0
        self.update_delay = 50  # milliseconds
        self.font_cache = {}
        
        # Font System
        self.system_fonts = self.find_common_fonts()
        self.current_font_path = self.system_fonts.get("Arial", None)
        
        # Configure Layout
        self.grid_columnconfigure(0, weight=0, minsize=420)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        
        self.setup_header()
        self.setup_sidebar()
        self.setup_canvas()
        self.setup_keyboard_shortcuts()
    
    def find_common_fonts(self):
        font_dict = {}
        win_path = r"C:\Windows\Fonts"
        targets = {
            "Arial": "arial.ttf", "Arial Bold": "arialbd.ttf",
            "Times New Roman": "times.ttf", "Times New Roman Bold": "timesbd.ttf",
            "Courier New": "cour.ttf", "Courier New Bold": "courbd.ttf",
            "Segoe UI": "segoeui.ttf", "Segoe UI Bold": "segoeuib.ttf",
            "Verdana": "verdana.ttf", "Calibri": "calibri.ttf", "Consolas": "consola.ttf"
        }
        for name, filename in targets.items():
            full_path = os.path.join(win_path, filename)
            if os.path.exists(full_path):
                font_dict[name] = full_path
        return font_dict
    
    def get_cached_font(self, font_path, size):
        """Cache fonts to avoid recreating them"""
        cache_key = (font_path, int(size))
        if cache_key not in self.font_cache:
            try:
                if font_path:
                    self.font_cache[cache_key] = ImageFont.truetype(font_path, int(size))
                else:
                    self.font_cache[cache_key] = ImageFont.truetype("arial.ttf", int(size))
            except:
                self.font_cache[cache_key] = ImageFont.load_default()
        return self.font_cache[cache_key]
    
    def setup_keyboard_shortcuts(self):
        self.bind('<Control-z>', lambda e: self.undo())
        self.bind('<Control-y>', lambda e: self.redo())
        self.bind('<Control-s>', lambda e: self.save_image())
        self.bind('<Control-o>', lambda e: self.load_image())
        self.bind('<space>', self.toggle_preview)
        self.bind('<Escape>', lambda e: self.cancel_edit())
        self.bind('<Return>', lambda e: self.commit_edit() if self.is_live_editing else None)
    
    def toggle_preview(self, event=None):
        if self.original_display is None or self.display_image is None:
            return
        
        self.showing_original = not self.showing_original
        
        if self.showing_original:
            self.temp_display = self.display_image
            self.display_image = self.original_display
            self.canvas_info.configure(text="👁️ ORIGINAL - Press SPACE to return")
        else:
            self.display_image = self.temp_display
            self.canvas_info.configure(text="✏️ Edited - Press SPACE for original")
        
        self.show_image()
        return "break"
    
    def cancel_edit(self):
        if self.is_live_editing:
            self.is_live_editing = False
            self.clean_image = None
            self.active_box_idx = -1
            self.btn_commit.configure(state="disabled", text="Select text", fg_color=COLORS['bg_light'])
            self.canvas_info.configure(text="❌ Cancelled")
            if self.history:
                self.display_image = self.history[-1].copy()
            self.show_image()
    
    def setup_header(self):
        header = ctk.CTkFrame(self, height=70, fg_color=COLORS['bg_medium'], corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        
        left_frame = ctk.CTkFrame(header, fg_color="transparent")
        left_frame.pack(side="left", padx=30, pady=15)
        
        ctk.CTkLabel(left_frame, text="✨ Image Text Editor Pro", 
                    font=("Segoe UI", 24, "bold"), text_color=COLORS['text_primary']).pack(side="left")
        ctk.CTkLabel(left_frame, text="Optimized Real-Time Engine",
                    font=("Segoe UI", 11), text_color=COLORS['text_secondary']).pack(side="left", padx=15)
        
        right_frame = ctk.CTkFrame(header, fg_color="transparent")
        right_frame.pack(side="right", padx=30)
        
        ctk.CTkButton(right_frame, text="📁 Open", command=self.load_image,
                     width=100, height=40, font=("Segoe UI", 12, "bold"),
                     fg_color=COLORS['primary'], hover_color=COLORS['primary_hover']).pack(side="left", padx=3)
        
        ctk.CTkButton(right_frame, text="💾 Save", command=self.save_image,
                     width=100, height=40, font=("Segoe UI", 12, "bold"),
                     fg_color=COLORS['success'], hover_color="#059669").pack(side="left", padx=3)
        
        ctk.CTkButton(right_frame, text="↩️", command=self.undo,
                     width=50, height=40, font=("Segoe UI", 14),
                     fg_color=COLORS['bg_light'], hover_color=COLORS['border']).pack(side="left", padx=2)
        
        ctk.CTkButton(right_frame, text="↪️", command=self.redo,
                     width=50, height=40, font=("Segoe UI", 14),
                     fg_color=COLORS['bg_light'], hover_color=COLORS['border']).pack(side="left", padx=2)
    
    def setup_sidebar(self):
        sidebar_container = ctk.CTkFrame(self, fg_color=COLORS['bg_dark'], corner_radius=0)
        sidebar_container.grid(row=1, column=0, sticky="nsew", padx=(15, 7), pady=15)
        
        self.sidebar = ctk.CTkScrollableFrame(sidebar_container, fg_color="transparent",
                                              scrollbar_button_color=COLORS['bg_light'],
                                              scrollbar_button_hover_color=COLORS['border'])
        self.sidebar.pack(fill="both", expand=True)
        
        # === DETECTION ===
        card1 = self.create_card_frame("🔍 AI Detection", COLORS['primary'])
        self.btn_detect = ctk.CTkButton(card1, text="Scan Document", command=self.start_detection,
                                       height=40, font=("Segoe UI", 12, "bold"),
                                       fg_color=COLORS['primary'], hover_color=COLORS['primary_hover'])
        self.btn_detect.pack(fill="x", pady=(0, 6))
        self.status_label = ctk.CTkLabel(card1, text="Ready", font=("Segoe UI", 9),
                                         text_color=COLORS['text_secondary'])
        self.status_label.pack()
        
        # === FONT ===
        card2 = self.create_card_frame("🎨 Font", COLORS['secondary'])
        self.detected_font_label = ctk.CTkLabel(card2, text="💡 Auto-match on select",
                                                font=("Segoe UI", 9), text_color=COLORS['text_secondary'], anchor="w")
        self.detected_font_label.pack(fill="x", pady=(0, 6))
        
        self.font_dropdown = ctk.CTkComboBox(card2, values=list(self.system_fonts.keys()),
                                            command=self.change_font, height=32, font=("Segoe UI", 10),
                                            button_color=COLORS['secondary'], button_hover_color=COLORS['primary'])
        self.font_dropdown.pack(fill="x", pady=(0, 5))
        
        ctk.CTkButton(card2, text="📂 Import", command=self.load_custom_font,
                     fg_color=COLORS['bg_light'], hover_color=COLORS['border'],
                     height=26, font=("Segoe UI", 9)).pack(fill="x")
        
        # === COLOR ===
        card3 = self.create_card_frame("🎨 Color", COLORS['warning'])
        color_buttons = ctk.CTkFrame(card3, fg_color="transparent")
        color_buttons.pack(fill="x", pady=(0, 6))
        
        self.btn_pick = ctk.CTkButton(color_buttons, text="🖌️ Pick", command=self.toggle_color_picker,
                                     fg_color=COLORS['bg_light'], hover_color=COLORS['border'],
                                     height=30, font=("Segoe UI", 10))
        self.btn_pick.pack(side="left", fill="x", expand=True, padx=(0, 3))
        
        ctk.CTkButton(color_buttons, text="🎨 Choose", command=self.open_color_dialog,
                     fg_color=COLORS['bg_light'], hover_color=COLORS['border'],
                     height=30, font=("Segoe UI", 10)).pack(side="right", fill="x", expand=True, padx=(3, 0))
        
        self.color_preview = ctk.CTkLabel(card3, text="#000000", fg_color="#000000",
                                         height=32, corner_radius=6, font=("Consolas", 10, "bold"), text_color="white")
        self.color_preview.pack(fill="x")
        
        # === TEXT ===
        card4 = self.create_card_frame("✏️ Text", COLORS['success'])
        self.entry_text = ctk.CTkEntry(card4, placeholder_text="Select text...",
                                      height=36, font=("Segoe UI", 11),
                                      border_color=COLORS['border'], fg_color=COLORS['bg_dark'])
        self.entry_text.pack(fill="x", pady=(0, 4))
        self.entry_text.bind("<KeyRelease>", self.on_text_change)
        
        self.memory_label = ctk.CTkLabel(card4, text="", font=("Segoe UI", 8), text_color=COLORS['warning'])
        self.memory_label.pack(fill="x")
        
        # === BASIC ===
        card5 = self.create_card_frame("📏 Basic", COLORS['primary'])
        self.add_slider(card5, "Size", 8, 200, 20, "font_size")
        self.add_slider(card5, "Spacing", -10, 20, 0, "spacing")
        self.add_slider(card5, "Rotation", -45, 45, 0, "rotation")
        self.add_slider(card5, "Y-Offset", -50, 50, 0, "offset_y")
        self.add_slider(card5, "X-Offset", -50, 50, 0, "offset_x")
        
        # === EFFECTS ===
        card6 = self.create_card_frame("✨ Effects", COLORS['secondary'])
        
        self.add_slider(card6, "Shadow X", -10, 10, 0, "shadow_x")
        self.add_slider(card6, "Shadow Y", -10, 10, 2, "shadow_y")
        self.add_slider(card6, "Shadow Blur", 0, 10, 3, "shadow_blur")
        self.add_slider(card6, "Shadow Opacity", 0, 255, 80, "shadow_opacity")
        
        self.add_slider(card6, "Outline Width", 0, 5, 0, "outline_width")
        self.add_slider(card6, "Outline Opacity", 0, 255, 255, "outline_opacity")
        
        self.add_slider(card6, "Noise", 0, 30, 0, "noise_level")
        
        # === ADVANCED ===
        card7 = self.create_card_frame("⚙️ Advanced", COLORS['warning'])
        self.add_slider(card7, "Opacity", 0, 255, 255, "opacity")
        self.add_slider(card7, "Blur", 0, 4, 0, "blur")
        
        # === COMMIT ===
        self.btn_commit = ctk.CTkButton(self.sidebar, text="✅ Apply (Enter)",
                                       command=self.commit_edit, fg_color=COLORS['success'],
                                       hover_color="#059669", state="disabled",
                                       height=42, font=("Segoe UI", 11, "bold"), corner_radius=8)
        self.btn_commit.pack(pady=8, fill="x")
        
        hints = ctk.CTkFrame(self.sidebar, fg_color=COLORS['bg_medium'], corner_radius=8)
        hints.pack(fill="x")
        ctk.CTkLabel(hints, text="⌨️ SPACE=Preview | ESC=Cancel",
                    font=("Segoe UI", 8), text_color=COLORS['text_secondary']).pack(pady=6)
    
    def create_card_frame(self, title, accent_color):
        card = ctk.CTkFrame(self.sidebar, fg_color=COLORS['bg_medium'], corner_radius=8)
        card.pack(fill="x", pady=(0, 8))
        
        header = ctk.CTkFrame(card, fg_color="transparent", height=28)
        header.pack(fill="x", padx=10, pady=(8, 6))
        
        ctk.CTkLabel(header, text="●", font=("Arial", 12), text_color=accent_color).pack(side="left", padx=(0, 6))
        ctk.CTkLabel(header, text=title, font=("Segoe UI", 10, "bold"),
                    text_color=COLORS['text_primary'], anchor="w").pack(side="left")
        
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="x", padx=10, pady=(0, 8))
        return content
    
    def add_slider(self, parent, label, min_val, max_val, default, attr_name):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=2)
        
        header = ctk.CTkFrame(container, fg_color="transparent")
        header.pack(fill="x", pady=(0, 1))
        
        ctk.CTkLabel(header, text=label, font=("Segoe UI", 8),
                    text_color=COLORS['text_secondary']).pack(side="left")
        val_lbl = ctk.CTkLabel(header, text=str(default), font=("Consolas", 8, "bold"),
                              text_color=COLORS['primary'])
        val_lbl.pack(side="right")
        
        def update_val(v):
            setattr(self, f"val_{attr_name}", v)
            fmt = f"{v:.1f}" if attr_name in ['blur', 'spacing', 'rotation'] else f"{int(v)}"
            val_lbl.configure(text=fmt)
            if self.is_live_editing:
                self.save_temp_state()
                self.schedule_update()
        
        slider = ctk.CTkSlider(container, from_=min_val, to=max_val,
                              number_of_steps=max(20, (max_val - min_val)), command=update_val,
                              height=12, button_color=COLORS['primary'],
                              button_hover_color=COLORS['primary_hover'], progress_color=COLORS['primary'])
        slider.set(default)
        slider.pack(fill="x")
        
        setattr(self, f"val_{attr_name}", default)
        setattr(self, f"slider_{attr_name}", slider)
    
    def schedule_update(self):
        """Debounced update for smooth real-time performance"""
        current_time = time.time() * 1000
        
        if not self.update_pending:
            self.update_pending = True
            delay = max(0, self.update_delay - (current_time - self.last_update_time))
            self.after(int(delay), self.execute_update)
    
    def execute_update(self):
        """Execute the actual update"""
        self.update_pending = False
        self.last_update_time = time.time() * 1000
        self.update_preview()
    
    def setup_canvas(self):
        canvas_container = ctk.CTkFrame(self, fg_color=COLORS['bg_medium'], corner_radius=10)
        canvas_container.grid(row=1, column=1, sticky="nsew", padx=(7, 15), pady=15)
        
        info_bar = ctk.CTkFrame(canvas_container, height=36, fg_color=COLORS['bg_dark'], corner_radius=8)
        info_bar.pack(fill="x", padx=8, pady=8)
        
        self.canvas_info = ctk.CTkLabel(info_bar, text="💡 Load image | SPACE=Preview",
                                        font=("Segoe UI", 9), text_color=COLORS['text_secondary'])
        self.canvas_info.pack(pady=8)
        
        canvas_frame = ctk.CTkFrame(canvas_container, fg_color=COLORS['bg_dark'], corner_radius=8)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        
        self.canvas = Canvas(canvas_frame, bg=COLORS['bg_dark'], highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.canvas.bind("<Button-1>", self.on_click)
    
    def save_temp_state(self):
        if self.active_box_idx == -1:
            return
        self.temp_edits[self.active_box_idx] = {
            'text': self.entry_text.get(), 'font_size': self.val_font_size,
            'spacing': self.val_spacing, 'offset_x': self.val_offset_x, 'offset_y': self.val_offset_y,
            'opacity': self.val_opacity, 'blur': self.val_blur, 'color': self.picked_color,
            'font': self.current_font_path, 'rotation': self.val_rotation,
            'shadow_x': self.val_shadow_x, 'shadow_y': self.val_shadow_y,
            'shadow_blur': self.val_shadow_blur, 'shadow_opacity': self.val_shadow_opacity,
            'outline_width': self.val_outline_width, 'outline_opacity': self.val_outline_opacity,
            'noise_level': self.val_noise_level
        }
        self.update_memory_indicator()
    
    def load_temp_state(self, idx):
        if idx in self.temp_edits:
            state = self.temp_edits[idx]
            self.entry_text.delete(0, "end")
            self.entry_text.insert(0, state['text'])
            
            for attr in ['font_size', 'spacing', 'offset_x', 'offset_y', 'opacity', 'blur',
                        'rotation', 'shadow_x', 'shadow_y', 'shadow_blur', 'shadow_opacity',
                        'outline_width', 'outline_opacity', 'noise_level']:
                getattr(self, f"slider_{attr}").set(state[attr])
            
            self.picked_color = state['color']
            hex_c = '#{:02x}{:02x}{:02x}'.format(*self.picked_color)
            self.update_color_preview(hex_c)
            self.current_font_path = state['font']
            return True
        return False
    
    def update_memory_indicator(self):
        count = len(self.temp_edits)
        self.memory_label.configure(text=f"⚠️ {count} unsaved" if count > 0 else "")
    
    def extract_text_color(self, box):
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        img_np = np.array(self.display_image)
        
        y1, y2 = max(0, y), min(img_np.shape[0], y + h)
        x1, x2 = max(0, x), min(img_np.shape[1], x + w)
        region = img_np[y1:y2, x1:x2]
        
        if region.size == 0:
            return (0, 0, 0)
        
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        threshold = np.percentile(gray, 30)
        text_mask = gray < threshold
        
        if np.any(text_mask) and len(region.shape) == 3:
            text_pixels = region[text_mask]
            median_color = np.median(text_pixels, axis=0).astype(int)
            return tuple(median_color)
        return (0, 0, 0)
    
    def detect_rotation(self, box):
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        img_np = np.array(self.display_image)
        
        y1, y2 = max(0, y), min(img_np.shape[0], y + h)
        x1, x2 = max(0, x), min(img_np.shape[1], x + w)
        region = img_np[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0
        
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=w//4, maxLineGap=5)
        
        if lines is not None and len(lines) > 0:
            angles = [np.degrees(np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0])) for line in lines]
            return np.median(angles)
        return 0
    
    def detect_font_properties(self, box):
        height = box['h']
        estimated_size = int(height * 0.75)
        
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        img_np = np.array(self.display_image)
        
        y1, y2 = max(0, y), min(img_np.shape[0], y + h)
        x1, x2 = max(0, x), min(img_np.shape[1], x + w)
        region = img_np[y1:y2, x1:x2]
        
        if len(region.shape) == 3:
            gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = region
        
        edges = cv2.Canny(gray_region, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
        is_bold = edge_density > 0.15
        
        detected_font = "Arial Bold" if is_bold and "Arial Bold" in self.system_fonts else "Arial"
        return estimated_size, detected_font, is_bold
    
    def change_font(self, choice):
        self.current_font_path = self.system_fonts[choice]
        if self.is_live_editing:
            self.save_temp_state()
            self.schedule_update()
    
    def load_custom_font(self):
        path = filedialog.askopenfilename(filetypes=[("Font", "*.ttf;*.otf")])
        if path:
            name = os.path.basename(path)
            self.system_fonts[name] = path
            self.font_dropdown.configure(values=list(self.system_fonts.keys()))
            self.font_dropdown.set(name)
            self.change_font(name)
    
    def open_color_dialog(self):
        color = colorchooser.askcolor(color=self.picked_color, title="Choose Color")
        if color[0]:
            self.picked_color = tuple(int(c) for c in color[0])
            hex_c = color[1]
            self.update_color_preview(hex_c)
            if self.is_live_editing:
                self.save_temp_state()
                self.schedule_update()
    
    def update_color_preview(self, hex_color):
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255
        text_color = "black" if luminance > 0.5 else "white"
        self.color_preview.configure(fg_color=hex_color, text=hex_color.upper(), text_color=text_color)
    
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            return
        
        self.cv_image = cv2.imread(path)
        if self.cv_image is None:
            return
        
        self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.display_image = Image.fromarray(self.cv_image)
        self.original_display = self.display_image.copy()
        self.clean_image = None
        self.boxes = []
        self.history = []
        self.redo_stack = []
        self.temp_edits = {}
        self.font_cache = {}
        self.is_live_editing = False
        self.canvas_info.configure(text="📷 Loaded - Scan document")
        self.update_memory_indicator()
        self.show_image()
    
    def start_detection(self):
        if self.display_image is None:
            messagebox.showwarning("No Image", "Load image first")
            return
        self.status_label.configure(text="🔄 Scanning...")
        self.btn_detect.configure(state="disabled", text="Scanning...")
        threading.Thread(target=self.run_ocr, daemon=True).start()
    
    def run_ocr(self):
        if self.reader is None:
            self.reader = easyocr.Reader(['en'], gpu=False)
        
        img_np = np.array(self.display_image)
        results = self.reader.readtext(img_np)
        
        self.boxes = []
        for (bbox, text, _) in results:
            (tl, tr, br, bl) = bbox
            self.boxes.append({
                'x': int(tl[0]), 'y': int(tl[1]),
                'w': int(br[0] - tl[0]), 'h': int(br[1] - tl[1]),
                'text': text
            })
        
        self.after(0, lambda: self.status_label.configure(text=f"✅ {len(self.boxes)} found"))
        self.after(0, lambda: self.btn_detect.configure(state="normal", text="Scan Document"))
        self.after(0, lambda: self.canvas_info.configure(text=f"🎯 {len(self.boxes)} detected - Click to edit"))
        self.after(0, self.show_image)
    
    def on_click(self, event):
        if not self.display_image:
            return
        
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        iw, ih = self.display_image.size
        scale = min(cw / iw, ch / ih)
        ox, oy = (cw - iw * scale) // 2, (ch - ih * scale) // 2
        
        mx, my = (event.x - ox) / scale, (event.y - oy) / scale
        
        if self.is_picking_color:
            if 0 <= mx < iw and 0 <= my < ih:
                pixel = self.display_image.getpixel((int(mx), int(my)))
                self.picked_color = pixel[:3]
                hex_c = '#{:02x}{:02x}{:02x}'.format(*self.picked_color)
                self.update_color_preview(hex_c)
                self.toggle_color_picker()
                if self.is_live_editing:
                    self.save_temp_state()
                    self.schedule_update()
            return
        
        if self.is_live_editing:
            self.commit_edit()
        
        for i, box in enumerate(self.boxes):
            if box['x'] < mx < box['x'] + box['w'] and box['y'] < my < box['y'] + box['h']:
                self.start_live_edit(i)
                return
    
    def start_live_edit(self, index):
        self.active_box_idx = index
        self.is_live_editing = True
        self.btn_commit.configure(state="normal", text="✅ Apply (Enter)", fg_color=COLORS['success'])
        
        box = self.boxes[index]
        
        if not self.load_temp_state(index):
            font_size, detected_font, is_bold = self.detect_font_properties(box)
            self.original_color = self.extract_text_color(box)
            rotation = self.detect_rotation(box)
            
            self.val_font_size = font_size
            self.slider_font_size.set(font_size)
            
            self.val_rotation = rotation
            self.slider_rotation.set(rotation)
            
            self.picked_color = self.original_color
            hex_c = '#{:02x}{:02x}{:02x}'.format(*self.original_color)
            self.update_color_preview(hex_c)
            
            self.font_dropdown.set(detected_font)
            self.current_font_path = self.system_fonts.get(detected_font)
            
            self.entry_text.delete(0, "end")
            self.entry_text.insert(0, box['text'])
            
            bold_str = "Bold" if is_bold else "Reg"
            rotation_str = f" • {rotation:.0f}°" if abs(rotation) > 1 else ""
            self.detected_font_label.configure(
                text=f"✓ {detected_font.split()[0]} {bold_str} • {font_size}pt{rotation_str}",
                text_color=COLORS['success']
            )
        else:
            self.detected_font_label.configure(text=f"📝 From memory", text_color=COLORS['warning'])
        
        img_np = np.array(self.display_image)
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        pad = 4
        mask[max(0, box['y'] - pad):box['y'] + box['h'] + pad,
             max(0, box['x'] - pad):box['x'] + box['w'] + pad] = 255
        self.clean_image = Image.fromarray(cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA))
        
        self.canvas_info.configure(text=f"✏️ Editing | SPACE=Preview | ESC=Cancel")
        self.update_preview()
    
    def on_text_change(self, event):
        if self.is_live_editing:
            self.save_temp_state()
            self.schedule_update()
    
    def update_preview(self):
        """Optimized real-time preview rendering"""
        if not self.clean_image or self.active_box_idx == -1:
            return
        
        base = self.clean_image.copy()
        box = self.boxes[self.active_box_idx]
        text = self.entry_text.get()
        
        if not text:
            self.display_image = base
            self.show_image()
            return
        
        txt_layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)
        
        # Get cached font
        font = self.get_cached_font(self.current_font_path, self.val_font_size)
        
        # TEXT SHADOW - Optimized
        if self.val_shadow_opacity > 10:
            shadow_layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            shadow_color = (0, 0, 0, int(self.val_shadow_opacity))
            
            x = box['x'] + self.val_offset_x + self.val_shadow_x
            y = box['y'] + self.val_offset_y + self.val_shadow_y
            
            shadow_draw.text((x, y), text, font=font, fill=shadow_color, spacing=self.val_spacing)
            
            if self.val_shadow_blur > 0.5:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=self.val_shadow_blur))
            txt_layer = Image.alpha_composite(txt_layer, shadow_layer)
        
        # TEXT OUTLINE - Optimized
        if self.val_outline_width > 0.5 and self.val_outline_opacity > 10:
            outline_color = (0, 0, 0, int(self.val_outline_opacity))
            x_base = box['x'] + self.val_offset_x
            y_base = box['y'] + self.val_offset_y
            
            # Simplified outline - 8 directions instead of full circle
            width = int(self.val_outline_width)
            for dx, dy in [(-width, -width), (0, -width), (width, -width),
                          (-width, 0), (width, 0),
                          (-width, width), (0, width), (width, width)]:
                draw.text((x_base + dx, y_base + dy), text, font=font, fill=outline_color, spacing=self.val_spacing)
        
        # MAIN TEXT
        fill = (self.picked_color[0], self.picked_color[1], self.picked_color[2], int(self.val_opacity))
        x = box['x'] + self.val_offset_x
        y = box['y'] + self.val_offset_y
        draw.text((x, y), text, font=font, fill=fill, spacing=self.val_spacing)
        
        # ROTATION - Optimized
        if abs(self.val_rotation) > 0.5:
            txt_bbox = txt_layer.getbbox()
            if txt_bbox:
                txt_cropped = txt_layer.crop(txt_bbox)
                txt_rotated = txt_cropped.rotate(-self.val_rotation, expand=True, resample=Image.BICUBIC)
                txt_layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
                paste_x = txt_bbox[0] - (txt_rotated.width - txt_cropped.width) // 2
                paste_y = txt_bbox[1] - (txt_rotated.height - txt_cropped.height) // 2
                txt_layer.paste(txt_rotated, (paste_x, paste_y), txt_rotated)
        
        # BLUR
        if self.val_blur > 0.2:
            txt_layer = txt_layer.filter(ImageFilter.GaussianBlur(radius=self.val_blur / 2))
        
        # NOISE - Optimized
        if self.val_noise_level > 5:
            # Only apply noise to text region
            txt_bbox = txt_layer.getbbox()
            if txt_bbox:
                x1, y1, x2, y2 = txt_bbox
                w, h = x2 - x1, y2 - y1
                noise = np.random.normal(0, self.val_noise_level / 2, (h, w, 4))
                noise = np.clip(noise, -50, 50).astype('int16')
                
                # Apply noise only to alpha channel area
                txt_array = np.array(txt_layer)
                region = txt_array[y1:y2, x1:x2]
                mask = region[:, :, 3] > 0
                region[mask] = np.clip(region[mask].astype('int16') + noise[mask], 0, 255).astype('uint8')
                txt_array[y1:y2, x1:x2] = region
                txt_layer = Image.fromarray(txt_array)
        
        base.paste(txt_layer, (0, 0), txt_layer)
        self.display_image = base
        self.show_image()
    
    def commit_edit(self):
        if not self.is_live_editing:
            return
        
        self.history.append(self.display_image.copy())
        self.redo_stack.clear()
        
        if self.active_box_idx in self.temp_edits:
            del self.temp_edits[self.active_box_idx]
        
        self.is_live_editing = False
        self.clean_image = None
        self.active_box_idx = -1
        self.detected_font_label.configure(text="💡 Auto-match on select", text_color=COLORS['text_secondary'])
        self.btn_commit.configure(state="disabled", text="Select text", fg_color=COLORS['bg_light'])
        self.canvas_info.configure(text="✅ Applied - SPACE to compare")
        self.update_memory_indicator()
        
        if not self.original_display:
            self.original_display = self.history[0].copy() if self.history else self.display_image.copy()
        self.show_image()
    
    def undo(self):
        if self.history:
            self.redo_stack.append(self.display_image.copy())
            self.display_image = self.history.pop()
            self.is_live_editing = False
            self.canvas_info.configure(text="↩️ Undo")
            self.show_image()
    
    def redo(self):
        if self.redo_stack:
            self.history.append(self.display_image.copy())
            self.display_image = self.redo_stack.pop()
            self.canvas_info.configure(text="↪️ Redo")
            self.show_image()
    
    def save_image(self):
        if self.display_image is None:
            messagebox.showwarning("No Image", "Load image first")
            return
        
        if self.temp_edits:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                f"{len(self.temp_edits)} unsaved edit(s).\n\nApply before saving?"
            )
            if response is None:
                return
            elif response and self.is_live_editing:
                self.commit_edit()
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
        )
        if path:
            self.display_image.save(path)
            messagebox.showinfo("Success", f"Saved!\n{os.path.basename(path)}")
    
    def show_image(self):
        if self.display_image is None:
            return
        
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            cw, ch = 1000, 700
        
        iw, ih = self.display_image.size
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        
        img = self.display_image.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self.canvas.create_image(ox + nw // 2, oy + nh // 2, image=self.tk_image)
        
        if not self.is_live_editing and not self.showing_original:
            for box in self.boxes:
                cx, cy = box['x'] * scale + ox, box['y'] * scale + oy
                bw, bh = box['w'] * scale, box['h'] * scale
                self.canvas.create_rectangle(cx, cy, cx + bw, cy + bh,
                                            outline=COLORS['success'], width=1, dash=(4, 4))
        elif self.active_box_idx != -1:
            box = self.boxes[self.active_box_idx]
            cx, cy = box['x'] * scale + ox, box['y'] * scale + oy
            bw, bh = box['w'] * scale, box['h'] * scale
            self.canvas.create_rectangle(cx - 1, cy - 1, cx + bw + 1, cy + bh + 1,
                                        outline=COLORS['primary'], width=2)
    
    def toggle_color_picker(self):
        self.is_picking_color = not self.is_picking_color
        
        if self.is_picking_color:
            self.btn_pick.configure(fg_color=COLORS['warning'], text="🎯 Click")
            self.canvas.configure(cursor="crosshair")
            self.status_label.configure(text="Click to pick")
        else:
            self.btn_pick.configure(fg_color=COLORS['bg_light'], text="🖌️ Pick")
            self.canvas.configure(cursor="hand2")
            self.status_label.configure(text="Ready")

if __name__ == "__main__":
    app = RealTimeEditor()
    app.mainloop()