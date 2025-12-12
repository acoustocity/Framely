import customtkinter as ctk
from tkinter import filedialog, Canvas, messagebox, colorchooser
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageFont, ImageDraw, ImageFilter
import easyocr
import threading
import os
import glob

# --- MODERN THEME CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Professional Color Palette
COLORS = {
    'primary': '#6366f1',
    'primary_hover': '#4f46e5',
    'secondary': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'bg_dark': '#0f172a',
    'bg_medium': '#1e293b',
    'bg_light': '#334155',
    'text_primary': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'border': '#475569'
}

class RealTimeEditor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Text Editor Pro")
        self.geometry("1600x950")
        self.configure(fg_color=COLORS['bg_dark'])
        
        # --- CORE STATE ---
        self.cv_image = None
        self.display_image = None
        self.clean_image = None
        self.boxes = []
        self.history = []
        self.redo_stack = []
        self.reader = None
        
        # --- EDITING STATE ---
        self.active_box_idx = -1
        self.is_live_editing = False
        self.picked_color = (0, 0, 0)
        self.is_picking_color = False
        self.original_color = (0, 0, 0)  # Store original text color
        self.temp_edits = {}  # Memory for unsaved edits
        
        # Font System
        self.system_fonts = self.find_common_fonts()
        self.current_font_path = self.system_fonts.get("Arial", None)
        
        # Configure Layout
        self.grid_columnconfigure(0, weight=0, minsize=380)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        
        self.setup_header()
        self.setup_sidebar()
        self.setup_canvas()
    
    def find_common_fonts(self):
        """Auto-detects common fonts on Windows"""
        font_dict = {}
        win_path = r"C:\Windows\Fonts"
        
        targets = {
            "Arial": "arial.ttf",
            "Arial Bold": "arialbd.ttf",
            "Times New Roman": "times.ttf",
            "Times New Roman Bold": "timesbd.ttf",
            "Courier New": "cour.ttf",
            "Courier New Bold": "courbd.ttf",
            "Segoe UI": "segoeui.ttf",
            "Segoe UI Bold": "segoeuib.ttf",
            "Verdana": "verdana.ttf",
            "Verdana Bold": "verdanab.ttf",
            "Calibri": "calibri.ttf",
            "Calibri Bold": "calibrib.ttf",
            "Consolas": "consola.ttf",
            "Tahoma": "tahoma.ttf"
        }
        
        for name, filename in targets.items():
            full_path = os.path.join(win_path, filename)
            if os.path.exists(full_path):
                font_dict[name] = full_path
        
        return font_dict
    
    def setup_header(self):
        """Modern header bar"""
        header = ctk.CTkFrame(self, height=70, fg_color=COLORS['bg_medium'], corner_radius=0)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_propagate(False)
        
        # Left side - Logo/Title
        left_frame = ctk.CTkFrame(header, fg_color="transparent")
        left_frame.pack(side="left", padx=30, pady=15)
        
        title_label = ctk.CTkLabel(
            left_frame, 
            text="✨ Image Text Editor Pro",
            font=("Segoe UI", 24, "bold"),
            text_color=COLORS['text_primary']
        )
        title_label.pack(side="left")
        
        subtitle = ctk.CTkLabel(
            left_frame,
            text="AI-Powered Text Detection & Editing",
            font=("Segoe UI", 11),
            text_color=COLORS['text_secondary']
        )
        subtitle.pack(side="left", padx=15)
        
        # Right side - Main actions
        right_frame = ctk.CTkFrame(header, fg_color="transparent")
        right_frame.pack(side="right", padx=30)
        
        self.btn_open = ctk.CTkButton(
            right_frame,
            text="📁 Open",
            command=self.load_image,
            width=110,
            height=40,
            font=("Segoe UI", 13, "bold"),
            fg_color=COLORS['primary'],
            hover_color=COLORS['primary_hover']
        )
        self.btn_open.pack(side="left", padx=5)
        
        self.btn_save = ctk.CTkButton(
            right_frame,
            text="💾 Save",
            command=self.save_image,
            width=100,
            height=40,
            font=("Segoe UI", 13, "bold"),
            fg_color=COLORS['success'],
            hover_color="#059669"
        )
        self.btn_save.pack(side="left", padx=5)
        
        # Undo/Redo buttons
        self.btn_undo = ctk.CTkButton(
            right_frame,
            text="↩️",
            command=self.undo,
            width=50,
            height=40,
            font=("Segoe UI", 18),
            fg_color=COLORS['bg_light'],
            hover_color=COLORS['border']
        )
        self.btn_undo.pack(side="left", padx=2)
        
        self.btn_redo = ctk.CTkButton(
            right_frame,
            text="↪️",
            command=self.redo,
            width=50,
            height=40,
            font=("Segoe UI", 18),
            fg_color=COLORS['bg_light'],
            hover_color=COLORS['border']
        )
        self.btn_redo.pack(side="left", padx=2)
    
    def setup_sidebar(self):
        """Modern sidebar with cards"""
        sidebar_container = ctk.CTkFrame(self, fg_color=COLORS['bg_dark'], corner_radius=0)
        sidebar_container.grid(row=1, column=0, sticky="nsew", padx=(15, 7), pady=15)
        
        self.sidebar = ctk.CTkScrollableFrame(
            sidebar_container,
            fg_color="transparent",
            scrollbar_button_color=COLORS['bg_light'],
            scrollbar_button_hover_color=COLORS['border']
        )
        self.sidebar.pack(fill="both", expand=True)
        
        # === CARD 1: AI DETECTION ===
        card1 = self.create_card_frame("🔍 AI Text Detection", COLORS['primary'])
        
        self.btn_detect = ctk.CTkButton(
            card1,
            text="Scan Document",
            command=self.start_detection,
            height=45,
            font=("Segoe UI", 13, "bold"),
            fg_color=COLORS['primary'],
            hover_color=COLORS['primary_hover']
        )
        self.btn_detect.pack(fill="x", pady=(0, 8))
        
        self.status_label = ctk.CTkLabel(
            card1,
            text="Ready to scan",
            font=("Segoe UI", 10),
            text_color=COLORS['text_secondary']
        )
        self.status_label.pack()
        
        # === CARD 2: FONT MATCHING ===
        card2 = self.create_card_frame("🎨 Smart Font Matching", COLORS['secondary'])
        
        # Compact detected info
        self.detected_font_label = ctk.CTkLabel(
            card2,
            text="💡 Select text to see properties",
            font=("Segoe UI", 11),
            text_color=COLORS['text_secondary'],
            anchor="w"
        )
        self.detected_font_label.pack(fill="x", pady=(0, 10))
        
        # Font selector
        ctk.CTkLabel(
            card2,
            text="Font Family",
            font=("Segoe UI", 10, "bold"),
            text_color=COLORS['text_primary'],
            anchor="w"
        ).pack(fill="x", pady=(0, 5))
        
        self.font_dropdown = ctk.CTkComboBox(
            card2,
            values=list(self.system_fonts.keys()),
            command=self.change_font,
            height=36,
            font=("Segoe UI", 11),
            button_color=COLORS['secondary'],
            button_hover_color=COLORS['primary']
        )
        self.font_dropdown.pack(fill="x", pady=(0, 8))
        
        ctk.CTkButton(
            card2,
            text="📂 Import Font",
            command=self.load_custom_font,
            fg_color=COLORS['bg_light'],
            hover_color=COLORS['border'],
            height=30,
            font=("Segoe UI", 10)
        ).pack(fill="x")
        
        # === CARD 3: COLOR TOOLS ===
        card3 = self.create_card_frame("🎨 Color Tools", COLORS['warning'])
        
        color_buttons = ctk.CTkFrame(card3, fg_color="transparent")
        color_buttons.pack(fill="x", pady=(0, 8))
        
        self.btn_pick = ctk.CTkButton(
            color_buttons,
            text="🖌️ Pick",
            command=self.toggle_color_picker,
            fg_color=COLORS['bg_light'],
            hover_color=COLORS['border'],
            height=34,
            font=("Segoe UI", 11)
        )
        self.btn_pick.pack(side="left", fill="x", expand=True, padx=(0, 4))
        
        self.btn_color_dialog = ctk.CTkButton(
            color_buttons,
            text="🎨 Choose",
            command=self.open_color_dialog,
            fg_color=COLORS['bg_light'],
            hover_color=COLORS['border'],
            height=34,
            font=("Segoe UI", 11)
        )
        self.btn_color_dialog.pack(side="right", fill="x", expand=True, padx=(4, 0))
        
        # Compact color preview
        self.color_preview = ctk.CTkLabel(
            card3,
            text="#000000",
            fg_color="#000000",
            height=38,
            corner_radius=6,
            font=("Consolas", 12, "bold"),
            text_color="white"
        )
        self.color_preview.pack(fill="x")
        
        # === CARD 4: TEXT EDITOR ===
        card4 = self.create_card_frame("✏️ Text Editor", COLORS['success'])
        
        self.entry_text = ctk.CTkEntry(
            card4,
            placeholder_text="Select text box to edit...",
            height=42,
            font=("Segoe UI", 12),
            border_color=COLORS['border'],
            fg_color=COLORS['bg_dark']
        )
        self.entry_text.pack(fill="x", pady=(0, 5))
        self.entry_text.bind("<KeyRelease>", self.on_text_change)
        
        # Memory indicator
        self.memory_label = ctk.CTkLabel(
            card4,
            text="",
            font=("Segoe UI", 9),
            text_color=COLORS['warning']
        )
        self.memory_label.pack(fill="x")
        
        # === CARD 5: FINE TUNING ===
        card5 = self.create_card_frame("⚙️ Fine Tuning", COLORS['primary'])
        
        self.add_modern_slider(card5, "Size", 8, 200, 20, "font_size")
        self.add_modern_slider(card5, "Spacing", -10, 20, 0, "spacing")
        self.add_modern_slider(card5, "Y-Offset", -50, 50, 0, "offset_y")
        self.add_modern_slider(card5, "X-Offset", -50, 50, 0, "offset_x")
        self.add_modern_slider(card5, "Opacity", 0, 255, 255, "opacity")
        self.add_modern_slider(card5, "Blur", 0, 4, 0, "blur")
        
        # === COMMIT BUTTON ===
        self.btn_commit = ctk.CTkButton(
            self.sidebar,
            text="✅ Apply Changes",
            command=self.commit_edit,
            fg_color=COLORS['success'],
            hover_color="#059669",
            state="disabled",
            height=48,
            font=("Segoe UI", 13, "bold"),
            corner_radius=8
        )
        self.btn_commit.pack(pady=12, fill="x")
    
    def create_card_frame(self, title, accent_color):
        """Creates a modern card container"""
        card = ctk.CTkFrame(self.sidebar, fg_color=COLORS['bg_medium'], corner_radius=10)
        card.pack(fill="x", pady=(0, 10))
        
        # Minimal header
        header = ctk.CTkFrame(card, fg_color="transparent", height=35)
        header.pack(fill="x", padx=12, pady=(10, 8))
        
        accent_dot = ctk.CTkLabel(header, text="●", font=("Arial", 16), text_color=accent_color)
        accent_dot.pack(side="left", padx=(0, 8))
        
        ctk.CTkLabel(
            header,
            text=title,
            font=("Segoe UI", 12, "bold"),
            text_color=COLORS['text_primary'],
            anchor="w"
        ).pack(side="left", fill="x")
        
        # Content area
        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="x", padx=12, pady=(0, 10))
        
        return content
    
    def add_modern_slider(self, parent, label, min_val, max_val, default, attr_name):
        """Compact modern slider"""
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", pady=4)
        
        # Compact header
        header = ctk.CTkFrame(container, fg_color="transparent")
        header.pack(fill="x", pady=(0, 3))
        
        ctk.CTkLabel(
            header,
            text=label,
            font=("Segoe UI", 10),
            text_color=COLORS['text_secondary']
        ).pack(side="left")
        
        val_lbl = ctk.CTkLabel(
            header,
            text=str(default),
            font=("Consolas", 10, "bold"),
            text_color=COLORS['primary']
        )
        val_lbl.pack(side="right")
        
        def update_val(v):
            setattr(self, f"val_{attr_name}", v)
            fmt = f"{v:.1f}" if attr_name in ['blur', 'spacing'] else f"{int(v)}"
            val_lbl.configure(text=fmt)
            if self.is_live_editing:
                self.save_temp_state()
                self.update_preview()
        
        slider = ctk.CTkSlider(
            container,
            from_=min_val,
            to=max_val,
            number_of_steps=(max_val - min_val) * 2,
            command=update_val,
            height=16,
            button_color=COLORS['primary'],
            button_hover_color=COLORS['primary_hover'],
            progress_color=COLORS['primary']
        )
        slider.set(default)
        slider.pack(fill="x")
        
        setattr(self, f"val_{attr_name}", default)
        setattr(self, f"slider_{attr_name}", slider)
    
    def setup_canvas(self):
        """Modern canvas area"""
        canvas_container = ctk.CTkFrame(self, fg_color=COLORS['bg_medium'], corner_radius=10)
        canvas_container.grid(row=1, column=1, sticky="nsew", padx=(7, 15), pady=15)
        
        # Minimal info bar
        info_bar = ctk.CTkFrame(canvas_container, height=40, fg_color=COLORS['bg_dark'], corner_radius=8)
        info_bar.pack(fill="x", padx=8, pady=8)
        
        self.canvas_info = ctk.CTkLabel(
            info_bar,
            text="💡 Load an image to get started",
            font=("Segoe UI", 10),
            text_color=COLORS['text_secondary']
        )
        self.canvas_info.pack(pady=8)
        
        # Canvas
        canvas_frame = ctk.CTkFrame(canvas_container, fg_color=COLORS['bg_dark'], corner_radius=8)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        
        self.canvas = Canvas(
            canvas_frame,
            bg=COLORS['bg_dark'],
            highlightthickness=0,
            cursor="hand2"
        )
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)
        self.canvas.bind("<Button-1>", self.on_click)
    
    # === MEMORY MANAGEMENT ===
    
    def save_temp_state(self):
        """Save current editing state to memory"""
        if self.active_box_idx == -1:
            return
        
        self.temp_edits[self.active_box_idx] = {
            'text': self.entry_text.get(),
            'font_size': self.val_font_size,
            'spacing': self.val_spacing,
            'offset_x': self.val_offset_x,
            'offset_y': self.val_offset_y,
            'opacity': self.val_opacity,
            'blur': self.val_blur,
            'color': self.picked_color,
            'font': self.current_font_path
        }
        self.update_memory_indicator()
    
    def load_temp_state(self, idx):
        """Restore saved editing state"""
        if idx in self.temp_edits:
            state = self.temp_edits[idx]
            self.entry_text.delete(0, "end")
            self.entry_text.insert(0, state['text'])
            
            self.slider_font_size.set(state['font_size'])
            self.slider_spacing.set(state['spacing'])
            self.slider_offset_x.set(state['offset_x'])
            self.slider_offset_y.set(state['offset_y'])
            self.slider_opacity.set(state['opacity'])
            self.slider_blur.set(state['blur'])
            
            self.picked_color = state['color']
            hex_c = '#{:02x}{:02x}{:02x}'.format(*self.picked_color)
            self.update_color_preview(hex_c)
            
            self.current_font_path = state['font']
            return True
        return False
    
    def update_memory_indicator(self):
        """Show unsaved changes count"""
        count = len(self.temp_edits)
        if count > 0:
            self.memory_label.configure(text=f"⚠️ {count} unsaved edit(s)")
        else:
            self.memory_label.configure(text="")
    
    # === COLOR EXTRACTION ===
    
    def extract_text_color(self, box):
        """Extract dominant color from text region"""
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        img_np = np.array(self.display_image)
        
        y1, y2 = max(0, y), min(img_np.shape[0], y + h)
        x1, x2 = max(0, x), min(img_np.shape[1], x + w)
        region = img_np[y1:y2, x1:x2]
        
        if region.size == 0:
            return (0, 0, 0)
        
        # Convert to grayscale to find dark pixels (text)
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray = region
        
        # Find darker pixels (likely text)
        threshold = np.percentile(gray, 30)
        text_mask = gray < threshold
        
        if np.any(text_mask) and len(region.shape) == 3:
            text_pixels = region[text_mask]
            # Get median color of text pixels
            median_color = np.median(text_pixels, axis=0).astype(int)
            return tuple(median_color)
        
        return (0, 0, 0)
    
    # === FONT DETECTION ===
    
    def detect_font_properties(self, box):
        """Estimate font size and boldness"""
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
    
    # === CORE FUNCTIONALITY ===
    
    def change_font(self, choice):
        self.current_font_path = self.system_fonts[choice]
        if self.is_live_editing:
            self.save_temp_state()
            self.update_preview()
    
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
                self.update_preview()
    
    def update_color_preview(self, hex_color):
        """Updates color preview with smart text color"""
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
        self.clean_image = None
        self.boxes = []
        self.history = []
        self.redo_stack = []
        self.temp_edits = {}
        self.is_live_editing = False
        self.canvas_info.configure(text="📷 Image loaded - Click 'Scan Document'")
        self.update_memory_indicator()
        self.show_image()
    
    def start_detection(self):
        if self.display_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
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
                'x': int(tl[0]),
                'y': int(tl[1]),
                'w': int(br[0] - tl[0]),
                'h': int(br[1] - tl[1]),
                'text': text
            })
        
        self.after(0, lambda: self.status_label.configure(text=f"✅ Found {len(self.boxes)} regions"))
        self.after(0, lambda: self.btn_detect.configure(state="normal", text="Scan Document"))
        self.after(0, lambda: self.canvas_info.configure(text=f"🎯 {len(self.boxes)} text detected - Click to edit"))
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
                    self.update_preview()
            return
        
        # Auto-commit when clicking new box
        if self.is_live_editing:
            self.commit_edit()
        
        for i, box in enumerate(self.boxes):
            if box['x'] < mx < box['x'] + box['w'] and box['y'] < my < box['y'] + box['h']:
                self.start_live_edit(i)
                return
    
    def start_live_edit(self, index):
        self.active_box_idx = index
        self.is_live_editing = True
        self.btn_commit.configure(state="normal", text="✅ Apply Changes", fg_color=COLORS['success'])
        
        box = self.boxes[index]
        
        # Check if we have saved state for this box
        if not self.load_temp_state(index):
            # Fresh edit - detect and match original properties
            font_size, detected_font, is_bold = self.detect_font_properties(box)
            self.original_color = self.extract_text_color(box)
            
            # Auto-set to match original
            self.val_font_size = font_size
            self.slider_font_size.set(font_size)
            
            self.picked_color = self.original_color
            hex_c = '#{:02x}{:02x}{:02x}'.format(*self.original_color)
            self.update_color_preview(hex_c)
            
            self.font_dropdown.set(detected_font)
            self.current_font_path = self.system_fonts.get(detected_font)
            
            self.entry_text.delete(0, "end")
            self.entry_text.insert(0, box['text'])
            
            bold_str = "Bold" if is_bold else "Regular"
            self.detected_font_label.configure(
                text=f"✓ {detected_font.split()[0]} {bold_str} • {font_size}pt • Auto-matched",
                text_color=COLORS['success']
            )
        else:
            # Restored from memory
            self.detected_font_label.configure(
                text=f"📝 Restored from memory",
                text_color=COLORS['warning']
            )
        
        # Inpainting
        img_np = np.array(self.display_image)
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        pad = 4
        mask[max(0, box['y'] - pad):box['y'] + box['h'] + pad,
             max(0, box['x'] - pad):box['x'] + box['w'] + pad] = 255
        self.clean_image = Image.fromarray(cv2.inpaint(img_np, mask, 3, cv2.INPAINT_TELEA))
        
        self.canvas_info.configure(text=f"✏️ Editing: '{box['text'][:30]}...'")
        self.update_preview()
    
    def on_text_change(self, event):
        if self.is_live_editing:
            self.save_temp_state()
            self.update_preview()
    
    def update_preview(self):
        if not self.clean_image or self.active_box_idx == -1:
            return
        
        base = self.clean_image.copy()
        txt_layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
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
            try:
                w = draw.textlength(char, font=font)
            except:
                bbox = draw.textbbox((0, 0), char, font=font)
                w = bbox[2] - bbox[0]
            x += w + self.val_spacing
        
        if self.val_blur > 0:
            txt_layer = txt_layer.filter(ImageFilter.GaussianBlur(radius=self.val_blur / 2))
        
        base.paste(txt_layer, (0, 0), txt_layer)
        self.display_image = base
        self.show_image()
    
    def commit_edit(self):
        if not self.is_live_editing:
            return
        
        # Save to history for undo
        self.history.append(self.display_image.copy())
        self.redo_stack.clear()
        
        # Clear temp memory for this box
        if self.active_box_idx in self.temp_edits:
            del self.temp_edits[self.active_box_idx]
        
        self.is_live_editing = False
        self.clean_image = None
        self.active_box_idx = -1
        self.detected_font_label.configure(text="💡 Select text to see properties", text_color=COLORS['text_secondary'])
        self.btn_commit.configure(state="disabled", text="Select text to edit", fg_color=COLORS['bg_light'])
        self.canvas_info.configure(text="✅ Changes applied")
        self.update_memory_indicator()
        self.show_image()
    
    def undo(self):
        if self.history:
            self.redo_stack.append(self.display_image.copy())
            self.display_image = self.history.pop()
            self.is_live_editing = False
            self.canvas_info.configure(text="↩️ Undo successful")
            self.show_image()
    
    def redo(self):
        if self.redo_stack:
            self.history.append(self.display_image.copy())
            self.display_image = self.redo_stack.pop()
            self.canvas_info.configure(text="↪️ Redo successful")
            self.show_image()
    
    def save_image(self):
        if self.display_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if self.temp_edits:
            response = messagebox.askyesnocancel(
                "Unsaved Changes",
                f"You have {len(self.temp_edits)} unsaved edit(s).\n\nDo you want to apply them before saving?"
            )
            if response is None:  # Cancel
                return
            elif response:  # Yes - apply changes
                if self.is_live_editing:
                    self.commit_edit()
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All Files", "*.*")]
        )
        if path:
            self.display_image.save(path)
            messagebox.showinfo("Success", f"Image saved!\n{os.path.basename(path)}")
    
    def show_image(self):
        if self.display_image is None:
            return
        
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10:
            cw, ch = 900, 700
        
        iw, ih = self.display_image.size
        scale = min(cw / iw, ch / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        
        img = self.display_image.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        self.canvas.create_image(ox + nw // 2, oy + nh // 2, image=self.tk_image)
        
        # Minimal box indicators
        if not self.is_live_editing:
            for box in self.boxes:
                cx, cy = box['x'] * scale + ox, box['y'] * scale + oy
                bw, bh = box['w'] * scale, box['h'] * scale
                # Subtle outline
                self.canvas.create_rectangle(
                    cx, cy, cx + bw, cy + bh,
                    outline=COLORS['success'], width=1, dash=(4, 4)
                )
        elif self.active_box_idx != -1:
            box = self.boxes[self.active_box_idx]
            cx, cy = box['x'] * scale + ox, box['y'] * scale + oy
            bw, bh = box['w'] * scale, box['h'] * scale
            # Active highlight
            self.canvas.create_rectangle(
                cx - 1, cy - 1, cx + bw + 1, cy + bh + 1,
                outline=COLORS['primary'], width=2
            )
    
    def toggle_color_picker(self):
        self.is_picking_color = not self.is_picking_color
        
        if self.is_picking_color:
            self.btn_pick.configure(fg_color=COLORS['warning'], text="🎯 Click")
            self.canvas.configure(cursor="crosshair")
            self.status_label.configure(text="Click image to pick color")
        else:
            self.btn_pick.configure(fg_color=COLORS['bg_light'], text="🖌️ Pick")
            self.canvas.configure(cursor="hand2")
            self.status_label.configure(text="Ready to scan")

if __name__ == "__main__":
    app = RealTimeEditor()
    app.mainloop()