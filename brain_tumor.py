import cv2
import numpy as np
import mediapipe as mp
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import math
import time
import os
from collections import deque
from tkinter import filedialog
import tkinter as tk
import glob

try:
    from ui_enhancements import EnhancedUI
    UI_ENHANCED = True
    print("Started")
except:
    UI_ENHANCED = False

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# GLB
try:
    import pygltflib
    GLTF_SUPPORT = True
except ImportError:
    GLTF_SUPPORT = False

# SETTINGS
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAM_WIDTH = 320
CAM_HEIGHT = 240
MRI_SIZE = 180
TARGET_FPS = 60

GESTURE_SMOOTH = 0.15
MIN_ZOOM = 0.5
MAX_ZOOM = 3.0

# COLORS
COLOR_BRAIN_NORMAL = (0.85, 0.75, 0.65, 0.75)
COLOR_BRAIN_XRAY = (0.85, 0.75, 0.65, 0.20)
COLOR_TUMOR = (0.95, 0.15, 0.1, 0.95)
COLOR_TUMOR_GLOW = (0.95, 0.15, 0.1, 0.4)
COLOR_HEALTHY = (0.2, 0.9, 0.3, 0.95)
COLOR_BG_NORMAL = (0.02, 0.02, 0.08)
COLOR_BG_XRAY = (0.3, 0.3, 0.35)

# PATHS
BRAIN_MODEL = "brain_model.glb"
AI_MODEL = "brain_tumor_model_lite.h5"
DATASET_PATH = "brain_mri_dataset"

class TumorAI:
    def __init__(self):
        self.model = None
        self.ready = False
        
        if not AI_AVAILABLE or not os.path.exists(AI_MODEL):
            print(" AI demo mode")
            return
            
        try:
            print(" Loading AI...")
            self.model = keras.models.load_model(AI_MODEL)
            self.ready = True
            print(" AI ready")
        except Exception as e:
            print(f"❌ AI error: {e}")
    
    def detect(self, mri_img):
        if not self.ready:
            return False, 0.3, (0, 0, 0)
        
        try:
            img = cv2.resize(mri_img, (128, 128))
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            pred = self.model.predict(img, verbose=0)[0][0]
            has_tumor = pred > 0.5
            
            gray = cv2.cvtColor(mri_img, cv2.COLOR_BGR2GRAY) if len(mri_img.shape) == 3 else mri_img
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours and has_tumor:
                largest = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest)
                
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    h, w = gray.shape
                    cx, cy = w//2, h//2
                
                h, w = gray.shape
                pos_x = (cx / w - 0.5) * 1.5
                pos_y = -(cy / h - 0.5) * 1.5
                pos_z = np.random.uniform(-0.2, 0.2)
                
                return has_tumor, float(pred), (pos_x, pos_y, pos_z)
            
            return has_tumor, float(pred), (0, 0, 0)
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return False, 0.0, (0, 0, 0)

class Brain3D:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.tumor_pos = None
        self.tumor_conf = 0.0
        self.has_tumor = False
        self.opacity = 0.75
        self.target_opacity = 0.75
        self.tumor_opacity = 0.95
        self.target_tumor_opacity = 0.95
        self.pulse = 0
        self.display_list = None
        self.quadric = None
        self.loaded = False
        
        
        self.base_rotation_x = 0
        self.base_rotation_y = 90  
        self.base_rotation_z = 0
        
    def load(self):
        if os.path.exists(BRAIN_MODEL) and GLTF_SUPPORT:
            self._load_glb()
        else:
            self._generate_brain()
    
    def _load_glb(self):
        try:
            print(" Loading brain...")
            glb = pygltflib.GLTF2().load(BRAIN_MODEL)
            blob = glb.binary_blob()
            
            verts = []
            faces = []
            offset = 0
            
            for mesh in glb.meshes:
                for prim in mesh.primitives:
                    if prim.attributes.POSITION is not None:
                        acc = glb.accessors[prim.attributes.POSITION]
                        view = glb.bufferViews[acc.bufferView]
                        
                        start = view.byteOffset
                        end = start + acc.count * 12
                        data = blob[start:end]
                        v = np.frombuffer(data, dtype=np.float32).reshape(-1, 3)
                        verts.extend(v)
                        
                        if prim.indices is not None:
                            idx_acc = glb.accessors[prim.indices]
                            idx_view = glb.bufferViews[idx_acc.bufferView]
                            
                            dtype = np.uint16 if idx_acc.componentType == 5123 else np.uint32
                            bpi = 2 if dtype == np.uint16 else 4
                            
                            idx_start = idx_view.byteOffset
                            idx_end = idx_start + idx_acc.count * bpi
                            idx_data = blob[idx_start:idx_end]
                            indices = np.frombuffer(idx_data, dtype=dtype)
                            
                            if len(indices) % 3 == 0:
                                f = indices.reshape(-1, 3) + offset
                                faces.extend(f)
                        
                        offset += len(v)
            
            self.vertices = np.array(verts)
            self.faces = np.array(faces)
            
            center = (np.min(self.vertices, axis=0) + np.max(self.vertices, axis=0)) / 2
            self.vertices -= center
            scale = 2.0 / np.max(np.max(self.vertices, axis=0) - np.min(self.vertices, axis=0))
            self.vertices *= scale
            
            max_idx = np.max(self.faces)
            if max_idx >= len(self.vertices):
                valid = [f for f in self.faces if all(i < len(self.vertices) for i in f)]
                self.faces = np.array(valid)
            
            self._create_display_list()
            print(f"Brain loaded: {len(self.vertices)} verts")
            self.loaded = True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self._generate_brain()
    
    def _generate_brain(self):
        print(" Generating brain...")
        segs = 30
        verts = []
        
        for i in range(segs + 1):
            lat = math.pi * (-0.5 + i / segs)
            cos_lat = math.cos(lat)
            sin_lat = math.sin(lat)
            
            for j in range(segs + 1):
                lon = 2 * math.pi * j / segs
                cos_lon = math.cos(lon)
                sin_lon = math.sin(lon)
                
                x = 1.2 * cos_lat * cos_lon
                y = 1.0 * cos_lat * sin_lon
                z = 0.9 * sin_lat
                
                bump = 0.03 * math.sin(lon * 8) * math.cos(lat * 6)
                verts.append((x + bump, y + bump, z + bump))
        
        faces = []
        for i in range(segs):
            for j in range(segs):
                f1 = i * (segs + 1) + j
                f2 = f1 + segs + 1
                faces.append([f1, f2, f1 + 1])
                faces.append([f2, f2 + 1, f1 + 1])
        
        self.vertices = np.array(verts)
        self.faces = np.array(faces)
        self._create_display_list()
        print(" Procedural brain ready")
        self.loaded = True
    
    def _create_display_list(self):
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for vi in face:
                if vi >= len(self.vertices):
                    continue
                v = self.vertices[vi]
                n = v / (np.linalg.norm(v) + 1e-6)
                glNormal3fv(n)
                glVertex3fv(v)
        glEnd()
        
        glEndList()
        
        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)
    
    def render(self, transform):
        if not self.loaded:
            return
        
        glPushMatrix()
        
        glTranslatef(0, 0, -5 * transform['zoom'])
        
        # SAME rotation as document
        glRotatef(self.base_rotation_x + transform['rotation_x'], 1, 0, 0)
        glRotatef(self.base_rotation_y + transform['rotation_y'], 0, 1, 0)
        glRotatef(self.base_rotation_z + transform['rotation_z'], 0, 0, 1)
        
        self.opacity += (self.target_opacity - self.opacity) * 0.15
        self.tumor_opacity += (self.target_tumor_opacity - self.tumor_opacity) * 0.15
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)
        
        glMaterialfv(GL_FRONT, GL_AMBIENT, [0.3, 0.25, 0.2, self.opacity])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, list(COLOR_BRAIN_NORMAL[:3]) + [self.opacity])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0.4, 0.4, 0.4, self.opacity])
        glMaterialf(GL_FRONT, GL_SHININESS, 20)
        
        if self.display_list:
            glCallList(self.display_list)
        
        if self.has_tumor and self.tumor_pos:
            self._render_tumor()
        elif not self.has_tumor and self.tumor_conf > 0:
            self._render_healthy()
        
        glDisable(GL_BLEND)
        glPopMatrix()
    
    def _render_tumor(self):
        self.pulse += 0.05
        pulse = 1.0 + 0.2 * math.sin(self.pulse * 3)
        
        glPushMatrix()
        glTranslatef(*self.tumor_pos)
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        
        size = 0.15 * pulse * self.tumor_conf
        
        glow_alpha = COLOR_TUMOR_GLOW[3] * (self.tumor_opacity / 0.95)
        glColor4f(COLOR_TUMOR_GLOW[0], COLOR_TUMOR_GLOW[1], COLOR_TUMOR_GLOW[2], glow_alpha)
        gluSphere(self.quadric, size * 2, 16, 16)
        
        core_alpha = self.tumor_opacity
        glColor4f(COLOR_TUMOR[0], COLOR_TUMOR[1], COLOR_TUMOR[2], core_alpha)
        gluSphere(self.quadric, size, 16, 16)
        
        glLineWidth(2)
        ring_alpha = 0.8 * (self.tumor_opacity / 0.95)
        glColor4f(1, 0.5, 0, ring_alpha)
        angle = (self.pulse * 30) % 360
        glRotatef(angle, 0, 1, 0)
        self._draw_wire_sphere(size * 2.5, 16, 16)
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        glPopMatrix()
    
    def _draw_wire_sphere(self, radius, slices, stacks):
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = radius * math.sin(lat0)
            r0 = radius * math.cos(lat0)
            
            glBegin(GL_LINE_LOOP)
            for j in range(slices):
                lng = 2 * math.pi * float(j) / slices
                x = math.cos(lng)
                y = math.sin(lng)
                glVertex3f(x * r0, y * r0, z0)
            glEnd()
    
    def _render_healthy(self):
        glPushMatrix()
        glTranslatef(0, 0.4, 0)
        
        glDisable(GL_LIGHTING)
        glColor4f(*COLOR_HEALTHY)
        gluSphere(self.quadric, 0.12, 16, 16)
        glEnable(GL_LIGHTING)
        glPopMatrix()
    
    def set_xray(self, active):
        if active:
            self.target_opacity = 0.20
            self.target_tumor_opacity = 0.45
        else:
            self.target_opacity = 0.75
            self.target_tumor_opacity = 0.95
    
    def update_tumor(self, has, conf, pos):
        self.has_tumor = has
        self.tumor_conf = conf
        self.tumor_pos = pos if has else None

class GestureControl:
    """SAME SIMPLE GESTURES as document - no changes"""
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_style = mp.solutions.drawing_styles
        
        self.rot_x = self.rot_y = self.rot_z = 0
        self.zoom = 1.5
        self.tgt_rot_x = self.tgt_rot_y = self.tgt_rot_z = 0
        self.tgt_zoom = 1.5
        self.rotating = False
        self.zooming = False
        self.xray = False
        self.last_results = None
        self.last_xray_state = False
        
    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        self.last_results = results
        
        if results.multi_hand_landmarks:
            hands = []
            for h in results.multi_hand_landmarks:
                hands.append([(lm.x, lm.y, lm.z) for lm in h.landmark])
            
            if len(hands) == 1:
                hand = hands[0]
                
                if self._is_fist(hand):
                    # X-RAY MODE
                    if not self.last_xray_state:
                        print(" X-RAY MODE")
                        self.last_xray_state = True
                    self.xray = True
                    self.rotating = False
                    self.zooming = False
                else:
                    # ROTATE
                    if self.last_xray_state:
                        print(" NORMAL MODE")
                        self.last_xray_state = False
                    
                    idx = hand[8]
                    x = idx[0] * CAM_WIDTH
                    y = idx[1] * CAM_HEIGHT
                    
                    self.tgt_rot_y = (x - CAM_WIDTH/2) * 0.6
                    self.tgt_rot_x = (y - CAM_HEIGHT/2) * 0.6
                    self.rotating = True
                    self.zooming = False
                    self.xray = False
                
            elif len(hands) == 2:
                # ZOOM
                if self.last_xray_state:
                    print(" NORMAL MODE")
                    self.last_xray_state = False
                
                dist = np.linalg.norm(np.array(hands[0][8][:2]) - np.array(hands[1][8][:2]))
                self.tgt_zoom = np.clip(dist * 3, MIN_ZOOM, MAX_ZOOM)
                self.zooming = True
                self.rotating = False
                self.xray = False
        else:
            if self.last_xray_state:
                print(" NORMAL MODE")
                self.last_xray_state = False
            
            self.rotating = False
            self.zooming = False
            self.xray = False
        
        self.rot_x += (self.tgt_rot_x - self.rot_x) * GESTURE_SMOOTH
        self.rot_y += (self.tgt_rot_y - self.rot_y) * GESTURE_SMOOTH
        self.rot_z += (self.tgt_rot_z - self.rot_z) * GESTURE_SMOOTH
        self.zoom += (self.tgt_zoom - self.zoom) * 0.1
        
        return {
            'rotation_x': self.rot_x,
            'rotation_y': self.rot_y,
            'rotation_z': self.rot_z,
            'zoom': self.zoom,
            'is_rotating': self.rotating,
            'is_zooming': self.zooming,
            'xray_mode': self.xray
        }
    
    def _is_fist(self, hand):
        palm = hand[0]
        tips = [hand[4], hand[8], hand[12], hand[16], hand[20]]
        distances = []
        for tip in tips:
            dist = math.sqrt((tip[0] - palm[0])**2 + (tip[1] - palm[1])**2)
            distances.append(dist)
        return sum(distances) / len(distances) < 0.15
    
    def draw_hands(self, frame):
        if self.last_results and self.last_results.multi_hand_landmarks:
            for hand in self.last_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_style.get_default_hand_landmarks_style(),
                    self.mp_style.get_default_hand_connections_style()
                )
        return frame

class ProfessionalUI:
    """Basic UI or Enhanced UI (if available)"""
    def __init__(self, screen):
        self.screen = screen
        self.w, self.h = screen.get_size()
        pygame.font.init()
        self.font_lg = pygame.font.Font(None, 36)
        self.font_md = pygame.font.Font(None, 28)
        self.font_sm = pygame.font.Font(None, 20)
        self.fps_hist = deque(maxlen=30)
        
        # Use enhanced UI if available
        if UI_ENHANCED:
            self.enhanced_ui = EnhancedUI(screen)
        
    def render(self, fps, tumor_data, gesture, cam_frame=None, mri_img=None):
        if UI_ENHANCED:
            # Use fancy UI
            self.enhanced_ui.render(fps, tumor_data, gesture, cam_frame, mri_img)
        else:
            # Use basic UI
            self._render_basic(fps, tumor_data, gesture, cam_frame, mri_img)
    
    def _render_basic(self, fps, tumor_data, gesture, cam_frame, mri_img):
        """Basic UI (same as document)"""
        # Top bar
        pygame.draw.rect(self.screen, (15, 25, 40), (0, 0, self.w, 50))
        
        title = self.font_md.render(" BRAIN TUMOR ANALYSIS", True, (240, 245, 255))
        self.screen.blit(title, (10, 12))
        
        self.fps_hist.append(fps)
        avg = sum(self.fps_hist) / len(self.fps_hist) if self.fps_hist else 0
        fps_col = (0,255,0) if avg > 50 else (255,255,0) if avg > 30 else (255,100,100)
        fps_txt = self.font_sm.render(f"{int(avg)} FPS", True, fps_col)
        self.screen.blit(fps_txt, (self.w - 70, 15))
        
        mode_x = self.w - 200
        if gesture['xray_mode']:
            mode = " X-RAY 20%"
            col = (255, 100, 255)
        elif gesture['is_rotating']:
            mode = " Rotating"
            col = (100, 200, 255)
        elif gesture['is_zooming']:
            mode = " Zooming"
            col = (255, 180, 100)
        else:
            mode = " Ready"
            col = (150, 150, 150)
        
        mode_txt = self.font_sm.render(mode, True, col)
        self.screen.blit(mode_txt, (mode_x, 15))
        
        self._draw_detection(tumor_data)
        
        if mri_img is not None:
            self._draw_mri(mri_img)
        
        if cam_frame is not None:
            self._draw_camera(cam_frame)
        
        self._draw_instructions()
    
    def _draw_detection(self, data):
        x, y = 20, 80
        w, h = 350, 240
        
        pygame.draw.rect(self.screen, (20, 30, 45), (x, y, w, h), border_radius=10)
        pygame.draw.rect(self.screen, (80, 120, 160), (x, y, w, h), 3, border_radius=10)
        
        title = self.font_md.render("DETECTION", True, (240, 245, 255))
        self.screen.blit(title, (x + 15, y + 15))
        
        has_tumor, conf = data['has_tumor'], data['confidence']
        
        if has_tumor:
            status = " TUMOR DETECTED"
            col = (255, 100, 100)
        else:
            status = " NO TUMOR"
            col = (100, 255, 150)
        
        status_txt = self.font_md.render(status, True, col)
        self.screen.blit(status_txt, (x + 15, y + 60))
        
        bar_y = y + 110
        bar_w = w - 30
        bar_h = 30
        
        pygame.draw.rect(self.screen, (40, 50, 60), (x + 15, bar_y, bar_w, bar_h), border_radius=5)
        
        fill = int(bar_w * conf)
        bar_col = (255, 80, 80) if conf > 0.7 else (255, 200, 80) if conf > 0.4 else (100, 200, 255)
        pygame.draw.rect(self.screen, bar_col, (x + 15, bar_y, fill, bar_h), border_radius=5)
        
        conf_txt = self.font_lg.render(f"{int(conf * 100)}%", True, (240, 245, 255))
        self.screen.blit(conf_txt, (x + 15, bar_y + 40))
        
        conf_label = self.font_sm.render("Confidence", True, (180, 190, 200))
        self.screen.blit(conf_label, (x + 15, bar_y + 80))
        
        if not data['has_mri']:
            upload = self.font_sm.render("Press U to upload MRI", True, (255, 200, 100))
            self.screen.blit(upload, (x + 15, bar_y + 110))
    
    def _draw_mri(self, mri):
        x = self.w - MRI_SIZE - 20
        y = self.h - MRI_SIZE - 20
        
        mri_small = cv2.resize(mri, (MRI_SIZE, MRI_SIZE))
        
        if len(mri_small.shape) == 2:
            mri_rgb = cv2.cvtColor(mri_small, cv2.COLOR_GRAY2RGB)
        else:
            mri_rgb = cv2.cvtColor(mri_small, cv2.COLOR_BGR2RGB)
        
        surf = pygame.surfarray.make_surface(mri_rgb.swapaxes(0, 1))
        
        pygame.draw.rect(self.screen, (100, 200, 255), (x-3, y-3, MRI_SIZE+6, MRI_SIZE+6), 3)
        self.screen.blit(surf, (x, y))
        
        label = self.font_sm.render("UPLOADED MRI", True, (200, 220, 255))
        self.screen.blit(label, (x, y - 25))
    
    def _draw_camera(self, frame):
        size = 120
        x = self.w - size - 20
        y = 70
        
        small = cv2.resize(frame, (size, size))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        
        pygame.draw.rect(self.screen, (80, 120, 160), (x-2, y-2, size+4, size+4), 2)
        self.screen.blit(surf, (x, y))
        
        label = self.font_sm.render("Gestures", True, (180, 190, 200))
        self.screen.blit(label, (x, y + size + 5))
    
    def _draw_instructions(self):
        y = self.h - 35
        
        inst = "Open Hand: Rotate 360° | Close Fist: X-Ray | 2 Hands: Zoom | U: Upload | ESC: Exit"
        txt = self.font_sm.render(inst, True, (180, 200, 220))
        rect = txt.get_rect(center=(self.w//2, y))
        self.screen.blit(txt, rect)

class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Brain Tumor 3D - Edge AI")
        
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, [2, 2, 3, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.4, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.9, 1])
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, WINDOW_WIDTH/WINDOW_HEIGHT, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        self.bg_color = COLOR_BG_NORMAL
        self.target_bg_color = COLOR_BG_NORMAL
        
        glClearColor(*self.bg_color, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        print("\n" + "="*70)
        print("  BRAIN TUMOR 3D ")
        print("="*70)
        print("\n Initializing...")
        
        self.ai = TumorAI()
        self.brain = Brain3D()
        self.brain.load()
        self.gesture = GestureControl()
        self.ui = ProfessionalUI(pygame.display.get_surface())
        
        self.cam = cv2.VideoCapture(0)
        if self.cam.isOpened():
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            print("Camera ready")
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        self.mri_image = None
        self.has_tumor = False
        self.confidence = 0.0
        self.tumor_pos = (0, 0, 0)
        
        self._load_sample_mri()
        
        print("✅ Ready!")
        print("\n Controls:")
        print("   • Open Hand = 360° Rotation")
        print("   • Close Fist = X-Ray Mode (20% + Gray BG)")
        print("   • 2 Hands = Zoom")
        print("   • U = Upload MRI")
        print("="*70 + "\n")
    
    def _load_sample_mri(self):
        if not os.path.exists(DATASET_PATH):
            return
        
        patterns = [
            os.path.join(DATASET_PATH, "yes", "*.jpg"),
            os.path.join(DATASET_PATH, "yes", "*.jpeg"),
            os.path.join(DATASET_PATH, "no", "*.jpg")
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                sample = files[0]
                print(f"📋 Loading sample: {os.path.basename(sample)}")
                
                img = cv2.imread(sample)
                if img is not None:
                    self.mri_image = img
                    self.has_tumor, self.confidence, self.tumor_pos = self.ai.detect(img)
                    self.brain.update_tumor(self.has_tumor, self.confidence, self.tumor_pos)
                    
                    status = "TUMOR" if self.has_tumor else "NORMAL"
                    print(f"Sample: {status} ({self.confidence*100:.1f}%)\n")
                    break
    
    def upload_mri(self):
        print("\n📁 Select MRI...")
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Select MRI",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")]
        )
        
        root.destroy()
        
        if file_path:
            print(f"Loading: {os.path.basename(file_path)}")
            
            img = cv2.imread(file_path)
            if img is not None:
                self.mri_image = img
                
                print("Analyzing...")
                self.has_tumor, self.confidence, self.tumor_pos = self.ai.detect(img)
                self.brain.update_tumor(self.has_tumor, self.confidence, self.tumor_pos)
                
                if self.has_tumor:
                    print(f"TUMOR! {self.confidence*100:.1f}%")
                else:
                    print(f" NORMAL {(1-self.confidence)*100:.1f}%")
                
                print("Done\n")
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_u:
                        self.upload_mri()
            
            cam_frame = None
            if self.cam.isOpened():
                ret, frame = self.cam.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    transform = self.gesture.process(frame)
                    cam_frame = self.gesture.draw_hands(frame)
                else:
                    transform = {'rotation_x': 0, 'rotation_y': 0, 'rotation_z': 0, 'zoom': 1.5,
                                'is_rotating': False, 'is_zooming': False, 'xray_mode': False}
            else:
                transform = {'rotation_x': 0, 'rotation_y': 0, 'rotation_z': 0, 'zoom': 1.5,
                            'is_rotating': False, 'is_zooming': False, 'xray_mode': False}
            
            self.brain.set_xray(transform['xray_mode'])
            
            self.target_bg_color = COLOR_BG_XRAY if transform['xray_mode'] else COLOR_BG_NORMAL
            
            self.bg_color = tuple(
                self.bg_color[i] + (self.target_bg_color[i] - self.bg_color[i]) * 0.1
                for i in range(3)
            )
            
            glClearColor(*self.bg_color, 1.0)
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            self.brain.render(transform)
            
            glDisable(GL_DEPTH_TEST)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            tumor_data = {
                'has_tumor': self.has_tumor,
                'confidence': self.confidence,
                'has_mri': self.mri_image is not None
            }
            
            self.ui.render(self.clock.get_fps(), tumor_data, transform, cam_frame, self.mri_image)
            
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glEnable(GL_DEPTH_TEST)
            
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)
        
        self.cleanup()
    
    def cleanup(self):
        if self.cam.isOpened():
            self.cam.release()
        pygame.quit()
        print("\n" + "="*70)
        print("  THANK YOU TO ALL ")
        print("="*70 + "\n")

if __name__ == "__main__":
    try:
        App().run()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()