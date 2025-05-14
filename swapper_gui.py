import sys
import os
import cv2
import numpy as np
import time
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QGridLayout, QScrollArea,
                            QStatusBar, QSlider, QMenu, QAction, QMessageBox, QInputDialog,
                            QComboBox, QCheckBox, QTabWidget, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QColor
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import pyaudio
import wave
import threading
import scipy.signal as signal

class FaceSwapperGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时换脸")
        self.setMinimumSize(1000, 600)
        
        # 初始化模型
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.swapper = get_model("./models/inswapper_128.onnx", 
                                download=False, 
                                providers=["CPUExecutionProvider"])
        
        # 存储人脸数据
        self.face_images = []
        self.face_features = []
        self.face_names = []
        self.selected_face_idx = -1
        self.current_source_face = None
        
        # 多人脸映射
        self.multi_face_enabled = False
        self.face_mapping = {}  # 目标脸索引 -> 源脸索引
        
        # 艺术滤镜
        self.current_filter = "无"
        self.available_filters = {
            "无": lambda img: img,
            "复古": self.apply_sepia,
            "素描": self.apply_sketch,
            "卡通": self.apply_cartoon,
            "边缘检测": self.apply_edge_detection,
            "浮雕": self.apply_emboss,
            "霓虹": self.apply_neon,
            "像素化": self.apply_pixelate
        }
        
        # AR贴纸
        self.stickers_enabled = False
        self.current_stickers = []  # 当前应用的贴纸列表 [(sticker_img, position_type), ...]
        self.sticker_positions = {
            "额头": "forehead",
            "鼻子": "nose",
            "眼睛": "eyes",
            "嘴巴": "mouth",
            "左耳": "left_ear",
            "右耳": "right_ear"
        }
        
        # 初始化摄像头
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 视频录制参数
        self.is_recording = False
        self.video_writer = None
        self.output_video_path = ""
        
        # 人脸交换参数
        self.blend_ratio = 1.0  # 1.0表示完全替换
        
        # 设置界面
        self.setup_ui()
        
        # 初始化状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("准备就绪")
        
        # 加载默认人脸和贴纸
        self.load_default_faces()
        self.load_default_stickers()

    def setup_ui(self):
        # 主布局
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：选项卡面板
        left_panel = QTabWidget()
        left_panel.setTabPosition(QTabWidget.North)
        
        # 人脸库选项卡
        face_tab = QWidget()
        face_layout = QVBoxLayout(face_tab)
        
        # 人脸网格容器（可滚动）
        self.face_scroll = QScrollArea()
        self.face_scroll.setWidgetResizable(True)
        self.face_container = QWidget()
        self.faces_grid = QGridLayout(self.face_container)
        self.face_scroll.setWidget(self.face_container)
        face_layout.addWidget(self.face_scroll)
        
        # 底部按钮
        buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("加载更多")
        self.load_button.setToolTip("从文件加载更多人脸图片")
        self.load_button.clicked.connect(self.load_more_faces)
        
        self.delete_button = QPushButton("删除选中")
        self.delete_button.setToolTip("删除当前选中的人脸")
        self.delete_button.clicked.connect(self.delete_selected_face)
        self.delete_button.setEnabled(False)
        
        self.rename_button = QPushButton("重命名")
        self.rename_button.setToolTip("为选中的人脸添加标签")
        self.rename_button.clicked.connect(self.rename_selected_face)
        self.rename_button.setEnabled(False)
        
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.delete_button)
        buttons_layout.addWidget(self.rename_button)
        face_layout.addLayout(buttons_layout)
        
        # 贴纸选项卡
        sticker_tab = QWidget()
        sticker_layout = QVBoxLayout(sticker_tab)
        
        # 贴纸启用复选框
        sticker_enable_layout = QHBoxLayout()
        self.sticker_checkbox = QCheckBox("启用贴纸")
        self.sticker_checkbox.setToolTip("启用/禁用AR贴纸功能")
        self.sticker_checkbox.stateChanged.connect(self.toggle_stickers)
        sticker_enable_layout.addWidget(self.sticker_checkbox)
        
        # 清除贴纸按钮
        self.clear_stickers_button = QPushButton("清除全部")
        self.clear_stickers_button.setToolTip("清除所有已添加的贴纸")
        self.clear_stickers_button.clicked.connect(self.clear_stickers)
        self.clear_stickers_button.setEnabled(False)
        sticker_enable_layout.addWidget(self.clear_stickers_button)
        
        sticker_layout.addLayout(sticker_enable_layout)
        
        # 贴纸列表
        self.sticker_list = QListWidget()
        self.sticker_list.setIconSize(QSize(40, 40))
        self.sticker_list.itemClicked.connect(self.on_sticker_clicked)
        self.sticker_list.setEnabled(False)
        sticker_layout.addWidget(self.sticker_list)
        
        # 添加两个选项卡
        left_panel.addTab(face_tab, "人脸库")
        # left_panel.addTab(sticker_tab, "AR贴纸")
        
        # 右侧：预览面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 预览标题
        preview_label = QLabel("实时预览")
        preview_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        right_layout.addWidget(preview_label)
        
        # 预览窗口
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet("background-color: #000;")
        right_layout.addWidget(self.preview_label)
        
        # 多人脸模式选择
        multi_face_layout = QHBoxLayout()
        self.multi_face_checkbox = QCheckBox("多人脸模式")
        self.multi_face_checkbox.setToolTip("启用多人脸同时换脸功能")
        self.multi_face_checkbox.stateChanged.connect(self.toggle_multi_face_mode)
        multi_face_layout.addWidget(self.multi_face_checkbox)
        
        # 清除映射按钮
        self.clear_mapping_button = QPushButton("清除映射")
        self.clear_mapping_button.setToolTip("清除所有人脸映射关系")
        self.clear_mapping_button.clicked.connect(self.clear_face_mapping)
        self.clear_mapping_button.setEnabled(False)
        multi_face_layout.addWidget(self.clear_mapping_button)
        
        right_layout.addLayout(multi_face_layout)
        
        # 混合比例滑块
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(QLabel("混合比例:"))
        self.blend_slider = QSlider(Qt.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(100)
        self.blend_slider.setToolTip("调整原始面孔和替换面孔的融合比例")
        self.blend_slider.valueChanged.connect(self.update_blend_ratio)
        blend_layout.addWidget(self.blend_slider)
        right_layout.addLayout(blend_layout)
        
        # 添加滤镜选择
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("艺术滤镜:"))
        self.filter_combo = QComboBox()
        for filter_name in self.available_filters.keys():
            self.filter_combo.addItem(filter_name)
        self.filter_combo.setToolTip("选择实时艺术滤镜效果")
        self.filter_combo.currentTextChanged.connect(self.change_filter)
        filter_layout.addWidget(self.filter_combo)
        right_layout.addLayout(filter_layout)
        
        # 操作按钮
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("开始换脸")
        self.start_button.setToolTip("开始/停止实时换脸")
        self.start_button.clicked.connect(self.toggle_face_swap)
        self.start_button.setEnabled(False)
        
        self.capture_button = QPushButton("拍照")
        self.capture_button.setToolTip("保存当前帧为图片")
        self.capture_button.clicked.connect(self.capture_frame)
        self.capture_button.setEnabled(False)
        
        self.record_button = QPushButton("开始录制")
        self.record_button.setToolTip("开始/停止视频录制")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.capture_button)
        control_layout.addWidget(self.record_button)
        
        right_layout.addLayout(control_layout)
        
        # 设置比例
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)
        
        self.setCentralWidget(central_widget)
        
        # 设置样式
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLabel {
                color: #333333;
            }
            QScrollArea {
                border: 1px solid #dddddd;
                background-color: #f9f9f9;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4a86e8;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            QCheckBox {
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QComboBox {
                border: 1px solid #bbb;
                border-radius: 3px;
                padding: 5px;
                min-width: 6em;
            }
            QComboBox::drop-down {
                border-left: 1px solid #bbb;
                width: 20px;
            }
            QTabWidget::pane {
                border: 1px solid #bbb;
                border-radius: 3px;
            }
            QTabBar::tab {
                background-color: #e8e8e8;
                border: 1px solid #bbb;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 6px 12px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom-color: #ffffff;
            }
            QListWidget {
                border: 1px solid #dddddd;
                background-color: #f9f9f9;
            }
            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid #eeeeee;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: #000000;
            }
        """)
    
    def toggle_multi_face_mode(self, state):
        self.multi_face_enabled = (state == Qt.Checked)
        self.clear_mapping_button.setEnabled(self.multi_face_enabled)
        
        if self.multi_face_enabled:
            self.statusBar.showMessage("已启用多人脸模式 - 点击预览窗口上的人脸进行映射")
            # 如果摄像头已开启，给预览标签添加点击事件
            if self.cap and self.cap.isOpened():
                self.preview_label.mousePressEvent = self.on_preview_click
        else:
            self.statusBar.showMessage("已禁用多人脸模式")
            # 移除预览标签的点击事件
            self.preview_label.mousePressEvent = None
            # 清除映射
            self.face_mapping = {}
    
    def clear_face_mapping(self):
        self.face_mapping = {}
        self.statusBar.showMessage("已清除所有人脸映射")
    
    def on_preview_click(self, event):
        if not hasattr(self, 'current_frame') or not hasattr(self, 'current_faces'):
            return
            
        # 获取点击位置
        preview_size = self.preview_label.size()
        frame_h, frame_w = self.current_frame.shape[:2]
        
        # 计算图像在预览窗口中的缩放比例和偏移
        preview_ar = preview_size.width() / preview_size.height()
        frame_ar = frame_w / frame_h
        
        if frame_ar > preview_ar:  # 图像宽度适配窗口宽度
            scale = preview_size.width() / frame_w
            offset_x = 0
            offset_y = (preview_size.height() - frame_h * scale) / 2
        else:  # 图像高度适配窗口高度
            scale = preview_size.height() / frame_h
            offset_x = (preview_size.width() - frame_w * scale) / 2
            offset_y = 0
        
        # 将点击坐标转换为原始图像坐标
        click_x = (event.x() - offset_x) / scale
        click_y = (event.y() - offset_y) / scale
        
        # 检查点击是否在某个人脸框内
        for i, face in enumerate(self.current_faces):
            box = face.bbox.astype(int)
            if (box[0] <= click_x <= box[2]) and (box[1] <= click_y <= box[3]):
                # 找到点击的人脸，弹出菜单选择源人脸
                self.show_face_mapping_menu(i, event.globalPos())
                break
    
    def show_face_mapping_menu(self, target_face_idx, position):
        if len(self.face_names) == 0:
            QMessageBox.warning(self, "警告", "请先添加源人脸图片")
            return
            
        # 创建菜单
        menu = QMenu(self)
        menu.setTitle(f"为目标人脸 {target_face_idx+1} 选择源人脸")
        
        # 添加源人脸选项
        for i, name in enumerate(self.face_names):
            action = QAction(f"{i+1}. {name}", self)
            action.triggered.connect(lambda checked, s=i, t=target_face_idx: self.map_faces(t, s))
            menu.addAction(action)
        
        # 如果已有映射，添加删除映射选项
        if target_face_idx in self.face_mapping:
            menu.addSeparator()
            remove_action = QAction("删除此映射", self)
            remove_action.triggered.connect(lambda: self.remove_face_mapping(target_face_idx))
            menu.addAction(remove_action)
        
        # 显示菜单
        menu.exec_(position)
    
    def map_faces(self, target_idx, source_idx):
        # 创建映射
        self.face_mapping[target_idx] = source_idx
        source_name = self.face_names[source_idx]
        self.statusBar.showMessage(f"已映射: 目标人脸 {target_idx+1} → 源人脸 '{source_name}'")
    
    def remove_face_mapping(self, target_idx):
        if target_idx in self.face_mapping:
            del self.face_mapping[target_idx]
            self.statusBar.showMessage(f"已删除目标人脸 {target_idx+1} 的映射")
    
    def update_blend_ratio(self, value):
        self.blend_ratio = value / 100.0
        self.statusBar.showMessage(f"混合比例: {value}%")
    
    def load_default_faces(self):
        # 加载默认目录中的人脸图片
        default_dir = "faces"
        if not os.path.exists(default_dir):
            os.makedirs(default_dir)
            print(f"创建了faces文件夹，请添加人脸图片到此文件夹")
            return
            
        # 打印当前路径和faces路径便于调试
        print(f"当前工作目录: {os.getcwd()}")
        print(f"faces目录路径: {os.path.abspath(default_dir)}")
        
        # 检查faces文件夹中的文件
        files = os.listdir(default_dir)
        print(f"faces文件夹中的文件: {files}")
        
        if files:
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(default_dir, filename)
                    print(f"尝试加载人脸图片: {filepath}")
                    # 使用文件名作为默认名称（不带扩展名）
                    name = os.path.splitext(filename)[0]
                    self.add_face_to_library(filepath, name)
        else:
            print("faces文件夹为空")
            
        # 如果没有默认人脸，加载source_face.jpg
        if len(self.face_images) == 0 and os.path.exists("source_face.jpg"):
            print("尝试加载默认人脸: source_face.jpg")
            self.add_face_to_library("source_face.jpg", "默认人脸")
    
    def add_face_to_library(self, image_path, name=None):
        try:
            # 处理中文路径问题
            if os.path.exists(image_path):
                # 使用numpy直接读取文件，避免cv2中文路径问题
                img = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            else:
                print(f"文件不存在: {image_path}")
                return
                
            if img is None:
                print(f"无法读取图片: {image_path}")
                return
                
            print(f"成功读取图片: {image_path}, 尺寸: {img.shape}")
            
            faces = self.app.get(img)
            if not faces:
                print(f"未检测到人脸: {image_path}")
                QMessageBox.warning(self, "警告", f"图片中未检测到人脸: {os.path.basename(image_path)}")
                return
                
            print(f"检测到人脸数量: {len(faces)}")
            
            # 获取第一个人脸
            face_feature = faces[0]
            
            # 如果没有提供名称，使用文件名（不带扩展名）
            if name is None:
                name = os.path.splitext(os.path.basename(image_path))[0]
                
            # 存储图片和特征
            self.face_images.append(img)
            self.face_features.append(face_feature)
            self.face_names.append(name)
            
            # 更新界面
            self.update_face_grid()
            self.statusBar.showMessage(f"成功添加人脸: {name}")
            print(f"成功添加人脸: {image_path}")
            
        except Exception as e:
            print(f"添加人脸失败: {image_path}")
            print(f"错误详情: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "错误", f"添加人脸失败: {str(e)}")
    
    def update_face_grid(self):
        # 清空网格
        while self.faces_grid.count():
            item = self.faces_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 重新填充网格
        for i, img in enumerate(self.face_images):
            # 创建容器小部件
            face_widget = QWidget()
            face_layout = QVBoxLayout(face_widget)
            face_layout.setSpacing(2)
            
            # 缩放图片
            h, w = img.shape[:2]
            aspect = w / h
            thumb_h = 120
            thumb_w = int(thumb_h * aspect)
            thumb = cv2.resize(img, (thumb_w, thumb_h))
            
            # 转换为QPixmap
            rgb_image = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            q_img = QImage(rgb_image.data, w, h, w * ch, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # 创建标签并添加点击事件
            label = QLabel()
            label.setPixmap(pixmap)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("padding: 5px; border: 2px solid transparent;")
            label.mousePressEvent = lambda event, idx=i: self.select_face(idx)
            
            # 如果是选中的人脸，添加高亮边框
            if i == self.selected_face_idx:
                label.setStyleSheet("padding: 5px; border: 2px solid #4a86e8;")
            
            # 添加名称标签
            name_label = QLabel(self.face_names[i])
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setStyleSheet("font-size: 12px;")
            
            # 添加到布局
            face_layout.addWidget(label)
            face_layout.addWidget(name_label)
            
            # 添加到网格
            row, col = divmod(i, 2)
            self.faces_grid.addWidget(face_widget, row, col)
    
    def select_face(self, idx):
        self.selected_face_idx = idx
        self.current_source_face = self.face_features[idx]
        self.update_face_grid()
        
        # 更新按钮状态
        self.start_button.setEnabled(True)
        self.delete_button.setEnabled(True)
        self.rename_button.setEnabled(True)
        
        self.statusBar.showMessage(f"已选择人脸: {self.face_names[idx]}")
    
    def delete_selected_face(self):
        if self.selected_face_idx == -1:
            return
            
        name = self.face_names[self.selected_face_idx]
        reply = QMessageBox.question(self, '确认删除', 
                                     f"确定要删除人脸 '{name}' 吗?",
                                     QMessageBox.Yes | QMessageBox.No, 
                                     QMessageBox.No)
                                     
        if reply == QMessageBox.Yes:
            # 删除选中的人脸
            del self.face_images[self.selected_face_idx]
            del self.face_features[self.selected_face_idx]
            del self.face_names[self.selected_face_idx]
            
            # 更新映射（如果有）
            for target_idx in list(self.face_mapping.keys()):
                if self.face_mapping[target_idx] == self.selected_face_idx:
                    # 删除受影响的映射
                    del self.face_mapping[target_idx]
                elif self.face_mapping[target_idx] > self.selected_face_idx:
                    # 更新索引大于删除索引的映射
                    self.face_mapping[target_idx] -= 1
            
            # 重置选择
            self.selected_face_idx = -1
            self.current_source_face = None
            
            # 更新UI
            self.update_face_grid()
            self.delete_button.setEnabled(False)
            self.rename_button.setEnabled(False)
            self.start_button.setEnabled(False)
            
            self.statusBar.showMessage(f"已删除人脸: {name}")
    
    def rename_selected_face(self):
        if self.selected_face_idx == -1:
            return
            
        current_name = self.face_names[self.selected_face_idx]
        new_name, ok = QInputDialog.getText(self, '重命名人脸', 
                                           '输入新名称:', text=current_name)
                                           
        if ok and new_name:
            self.face_names[self.selected_face_idx] = new_name
            self.update_face_grid()
            self.statusBar.showMessage(f"已重命名人脸: {new_name}")
    
    def load_more_faces(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        
        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            print(f"选择的文件: {filenames}")
            for filename in filenames:
                print(f"尝试加载: {filename}")
                self.add_face_to_library(filename)
    
    def toggle_face_swap(self):
        if self.timer.isActive():
            self.timer.stop()
            if self.cap:
                self.cap.release()
            self.cap = None
            self.start_button.setText("开始换脸")
            self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("预览已停止")
            self.capture_button.setEnabled(False)
            self.record_button.setEnabled(False)
            
            # 如果正在录制，停止录制
            if self.is_recording:
                self.toggle_recording()
                
            self.statusBar.showMessage("换脸已停止")
            
            # 移除预览标签的点击事件
            self.preview_label.mousePressEvent = None
        else:
            if self.selected_face_idx == -1 and not self.multi_face_enabled:
                QMessageBox.warning(self, "警告", "请先选择一个源人脸")
                return
            
            # 尝试不同的摄像头索引
            camera_indices = [0, 1, -1]
            for idx in camera_indices:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    break
            
            if not self.cap or not self.cap.isOpened():
                print("无法打开摄像头，请检查设备连接")
                self.preview_label.setText("无法打开摄像头，请检查设备连接")
                QMessageBox.critical(self, "错误", "无法打开摄像头，请检查设备连接")
                return
            
            print(f"成功打开摄像头")
            self.timer.start(30)  # 约30FPS
            self.start_button.setText("停止换脸")
            self.capture_button.setEnabled(True)
            self.record_button.setEnabled(True)
            
            if self.multi_face_enabled:
                self.statusBar.showMessage("多人脸换脸已开始 - 点击预览中的人脸进行映射")
                # 给预览标签添加点击事件
                self.preview_label.mousePressEvent = self.on_preview_click
            else:
                self.statusBar.showMessage(f"换脸已开始 - 使用人脸: {self.face_names[self.selected_face_idx]}")
    
    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            print("无法读取摄像头画面")
            return
        
        # 水平翻转图像（镜像），使其更直观
        frame = cv2.flip(frame, 1)
        
        display_frame = frame.copy()
        
        # 进行换脸
        try:
            target_faces = self.app.get(frame)
            self.current_faces = target_faces  # 保存当前帧的人脸，用于点击映射
            
            if target_faces:
                # 多人脸模式
                if self.multi_face_enabled:
                    for i, target_face in enumerate(target_faces):
                        # 显示每个人脸的框和索引
                        box = target_face.bbox.astype(int)
                        cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), 
                                     (0, 255, 0), 2)
                        # 显示人脸索引
                        cv2.putText(display_frame, f"Face {i+1}", (box[0], box[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 如果有映射，执行换脸
                        if i in self.face_mapping:
                            source_idx = self.face_mapping[i]
                            source_face = self.face_features[source_idx]
                            
                            # 根据混合比例执行换脸
                            if self.blend_ratio >= 0.99:
                                display_frame = self.swapper.get(display_frame, target_face, source_face, paste_back=True)
                            elif self.blend_ratio > 0:
                                swapped = self.swapper.get(frame.copy(), target_face, source_face, paste_back=True)
                                # 在换脸区域进行混合
                                box = target_face.bbox.astype(int)
                                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                                
                                # 扩大一点区域以获得更好的混合效果
                                margin = 30
                                x1 = max(0, x1 - margin)
                                y1 = max(0, y1 - margin)
                                x2 = min(frame.shape[1], x2 + margin)
                                y2 = min(frame.shape[0], y2 + margin)
                                
                                # 只混合人脸区域
                                display_frame[y1:y2, x1:x2] = cv2.addWeighted(
                                    display_frame[y1:y2, x1:x2], 
                                    1 - self.blend_ratio,
                                    swapped[y1:y2, x1:x2], 
                                    self.blend_ratio, 
                                    0
                                )
                            
                            # 在人脸框上显示源人脸名称
                            name = self.face_names[source_idx]
                            cv2.putText(display_frame, name, (box[0], box[3] + 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
                            # 添加AR贴纸
                            if self.stickers_enabled and self.current_stickers:
                                self.apply_stickers(display_frame, target_face)
                
                # 单人脸模式
                else:
                    target_face = target_faces[0]
                    box = target_face.bbox.astype(int)
                    
                    # 使用混合比例
                    if self.blend_ratio >= 0.99:  # 近似为1时，直接完全替换
                        display_frame = self.swapper.get(frame, target_face, self.current_source_face, paste_back=True)
                    elif self.blend_ratio > 0:
                        # 获取换脸结果
                        swapped_frame = self.swapper.get(frame, target_face, self.current_source_face, paste_back=True)
                        # 混合原始帧和换脸帧
                        display_frame = cv2.addWeighted(frame, 1 - self.blend_ratio, swapped_frame, self.blend_ratio, 0)
                    
                    # 在帧上显示检测到的人脸
                    cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    
                    # 添加AR贴纸
                    if self.stickers_enabled and self.current_stickers:
                        self.apply_stickers(display_frame, target_face)
            else:
                # 如果没检测到人脸，在帧上显示提示
                cv2.putText(display_frame, "未检测到人脸", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"换脸失败: {str(e)}")
            cv2.putText(display_frame, f"换脸失败: {str(e)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 应用艺术滤镜
        if self.current_filter != "无":
            display_frame = self.available_filters[self.current_filter](display_frame)
        
        # 如果正在录制，写入帧
        if self.is_recording and self.video_writer:
            self.video_writer.write(display_frame)
            
            # 添加录制指示器
            radius = 20
            center = (radius + 10, radius + 10)
            # 闪烁效果 - 使用时间的奇偶秒
            if int(time.time()) % 2 == 0:
                cv2.circle(display_frame, center, radius, (0, 0, 255), -1)
            cv2.putText(display_frame, "REC", (center[0] + radius, center[1] + 7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 转换帧并显示
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        q_img = QImage(rgb_frame.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 缩放以适应标签
        pixmap = pixmap.scaled(self.preview_label.width(), self.preview_label.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)
        
        # 保存当前帧用于可能的截图
        self.current_frame = display_frame
    
    def capture_frame(self):
        if not hasattr(self, 'current_frame') or self.current_frame is None:
            return
            
        # 创建保存目录
        save_dir = "captures"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 生成文件名 (使用当前时间戳)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{save_dir}/capture_{timestamp}.jpg"
        
        # 保存图片
        cv2.imwrite(filename, self.current_frame)
        self.statusBar.showMessage(f"已保存截图: {filename}")
        
        # 显示确认消息
        QMessageBox.information(self, "保存成功", f"已保存截图到: {filename}")
    
    def toggle_recording(self):
        if self.is_recording:
            # 停止录制
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            
            self.record_button.setText("开始录制")
            self.statusBar.showMessage(f"视频录制已停止: {self.output_video_path}")
            
            # 显示确认消息
            QMessageBox.information(self, "录制完成", f"视频已保存到: {self.output_video_path}")
        else:
            # 开始录制
            # 创建保存目录
            save_dir = "videos"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            # 生成文件名 (使用当前时间戳)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.output_video_path = f"{save_dir}/video_{timestamp}.mp4"
            
            # 获取当前帧的大小
            if hasattr(self, 'current_frame'):
                height, width = self.current_frame.shape[:2]
            else:
                # 默认大小
                width, height = 640, 480
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4格式
            self.video_writer = cv2.VideoWriter(
                self.output_video_path, fourcc, 20.0, (width, height))
            
            self.is_recording = True
            self.record_button.setText("停止录制")
            self.statusBar.showMessage("视频录制中...")
    
    def closeEvent(self, event):
        # 停止摄像头和录制
        if self.cap and self.cap.isOpened():
            self.cap.release()
            
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            
        event.accept()

    def change_filter(self, filter_name):
        self.current_filter = filter_name
        self.statusBar.showMessage(f"已切换滤镜: {filter_name}")

    def apply_sepia(self, img):
        # 复古棕褐色滤镜
        sepia_filter = np.array([[0.393, 0.769, 0.189],
                               [0.349, 0.686, 0.168],
                               [0.272, 0.534, 0.131]])
        sepia_img = cv2.transform(img, sepia_filter)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return sepia_img

    def apply_sketch(self, img):
        # 素描效果
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv_gray = 255 - gray
        blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blurred, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def apply_cartoon(self, img):
        # 卡通效果
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon

    def apply_edge_detection(self, img):
        # 边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def apply_emboss(self, img):
        # 浮雕效果
        kernel = np.array([[0, -1, -1],
                          [1, 0, -1],
                          [1, 1, 0]])
        emboss = cv2.filter2D(img, -1, kernel) + 128
        return emboss

    def apply_neon(self, img):
        # 霓虹效果
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        edges = cv2.divide(gray, blurred, scale=256)
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
        edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)[1]
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 添加霓虹色彩
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # 随机调整色相
        h = (h + np.random.randint(0, 180)) % 180
        s = np.clip(s * 1.5, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 结合边缘和颜色
        result = cv2.bitwise_and(colored, edges)
        return result

    def apply_pixelate(self, img):
        # 像素化效果
        height, width = img.shape[:2]
        
        # 定义像素块大小
        block_size = 15
        
        # 缩小图像
        small = cv2.resize(img, (width // block_size, height // block_size),
                         interpolation=cv2.INTER_LINEAR)
        
        # 放大回原始尺寸
        pixelated = cv2.resize(small, (width, height), 
                             interpolation=cv2.INTER_NEAREST)
        
        return pixelated

    def toggle_stickers(self, state):
        self.stickers_enabled = (state == Qt.Checked)
        self.sticker_list.setEnabled(self.stickers_enabled)
        self.clear_stickers_button.setEnabled(self.stickers_enabled)
        
        if self.stickers_enabled:
            self.statusBar.showMessage("已启用AR贴纸功能")
        else:
            self.statusBar.showMessage("已禁用AR贴纸功能")
            self.clear_stickers()
    
    def clear_stickers(self):
        self.current_stickers = []
        self.statusBar.showMessage("已清除所有贴纸")
    
    def load_default_stickers(self):
        # 创建贴纸目录
        stickers_dir = "stickers"
        if not os.path.exists(stickers_dir):
            os.makedirs(stickers_dir)
            print(f"创建了stickers文件夹，请添加贴纸图片到此文件夹")
            return
        
        # 加载贴纸
        if os.path.exists(stickers_dir):
            sticker_files = [f for f in os.listdir(stickers_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for sticker_file in sticker_files:
                sticker_path = os.path.join(stickers_dir, sticker_file)
                self.add_sticker_to_list(sticker_path)
    
    def add_sticker_to_list(self, sticker_path):
        try:
            # 读取贴纸图片
            img = np.fromfile(sticker_path, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)  # 保留透明通道
            
            if img is None:
                print(f"无法读取贴纸: {sticker_path}")
                return
            
            # 创建缩略图
            name = os.path.splitext(os.path.basename(sticker_path))[0]
            icon_img = cv2.resize(img, (40, 40))
            
            # 转换为QIcon
            if img.shape[2] == 4:  # 有透明通道
                qimg = QImage(icon_img.data, icon_img.shape[1], icon_img.shape[0], 
                             QImage.Format_RGBA8888)
            else:
                # 转换为RGB
                icon_img = cv2.cvtColor(icon_img, cv2.COLOR_BGR2RGB)
                qimg = QImage(icon_img.data, icon_img.shape[1], icon_img.shape[0], 
                             icon_img.shape[1] * 3, QImage.Format_RGB888)
            
            pixmap = QPixmap.fromImage(qimg)
            icon = QIcon(pixmap)
            
            # 添加到列表
            item = QListWidgetItem(icon, name)
            item.setData(Qt.UserRole, sticker_path)  # 存储贴纸路径
            self.sticker_list.addItem(item)
            
        except Exception as e:
            print(f"添加贴纸失败: {sticker_path}")
            print(f"错误详情: {str(e)}")
    
    def on_sticker_clicked(self, item):
        if not self.stickers_enabled:
            return
            
        sticker_path = item.data(Qt.UserRole)
        
        # 创建位置选择菜单
        menu = QMenu(self)
        menu.setTitle("选择贴纸位置")
        
        for pos_name in self.sticker_positions.keys():
            action = QAction(pos_name, self)
            pos_type = self.sticker_positions[pos_name]
            action.triggered.connect(lambda checked, p=sticker_path, t=pos_type: 
                                    self.add_sticker(p, t))
            menu.addAction(action)
        
        # 显示菜单
        menu.exec_(self.sticker_list.mapToGlobal(self.sticker_list.visualItemRect(item).bottomLeft()))
    
    def add_sticker(self, sticker_path, position_type):
        try:
            # 读取贴纸图片（带透明通道）
            img = np.fromfile(sticker_path, dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"无法读取贴纸: {sticker_path}")
                return
                
            # 将贴纸和位置类型添加到当前贴纸列表
            self.current_stickers.append((img, position_type))
            
            # 显示状态信息
            sticker_name = os.path.splitext(os.path.basename(sticker_path))[0]
            pos_name = [k for k, v in self.sticker_positions.items() if v == position_type][0]
            self.statusBar.showMessage(f"已添加贴纸 '{sticker_name}' 到位置: {pos_name}")
            
        except Exception as e:
            print(f"添加贴纸失败: {sticker_path}")
            print(f"错误详情: {str(e)}")

    def apply_stickers(self, frame, face):
        # 获取人脸关键点
        landmarks = face.landmark_2d_106
        if landmarks is None or len(landmarks) < 106:
            return
        
        landmarks = landmarks.astype(np.int32)
        
        # 确定各部位中心点和大小
        # 额头位置（使用眉毛上方）
        forehead_center = (int((landmarks[33][0] + landmarks[38][0]) / 2), 
                          int(landmarks[33][1] - (landmarks[66][1] - landmarks[33][1]) * 0.5))
        forehead_size = int((landmarks[38][0] - landmarks[33][0]) * 1.2)
        
        # 鼻子位置
        nose_center = (int(landmarks[51][0]), int(landmarks[51][1]))
        nose_size = int((landmarks[54][0] - landmarks[48][0]) * 0.8)
        
        # 眼睛位置（两眼中心）
        left_eye = (int((landmarks[60][0] + landmarks[61][0]) / 2), 
                   int((landmarks[60][1] + landmarks[61][1]) / 2))
        right_eye = (int((landmarks[68][0] + landmarks[69][0]) / 2), 
                    int((landmarks[68][1] + landmarks[69][1]) / 2))
        eyes_center = (int((left_eye[0] + right_eye[0]) / 2), 
                      int((left_eye[1] + right_eye[1]) / 2))
        eyes_size = int((right_eye[0] - left_eye[0]) * 1.5)
        
        # 嘴巴位置
        mouth_center = (int((landmarks[76][0] + landmarks[82][0]) / 2), 
                       int((landmarks[76][1] + landmarks[82][1]) / 2))
        mouth_size = int((landmarks[82][0] - landmarks[76][0]) * 1.2)
        
        # 左耳和右耳位置（估计位置，实际关键点可能没有耳朵）
        face_width = int(face.bbox[2] - face.bbox[0])
        left_ear_center = (int(face.bbox[0] - face_width * 0.1), eyes_center[1])
        right_ear_center = (int(face.bbox[2] + face_width * 0.1), eyes_center[1])
        ear_size = int(face_width * 0.2)
        
        # 应用每个贴纸
        for sticker_img, position in self.current_stickers:
            if position == "forehead":
                self.add_sticker_to_frame(frame, sticker_img, forehead_center, forehead_size)
            elif position == "nose":
                self.add_sticker_to_frame(frame, sticker_img, nose_center, nose_size)
            elif position == "eyes":
                self.add_sticker_to_frame(frame, sticker_img, eyes_center, eyes_size)
            elif position == "mouth":
                self.add_sticker_to_frame(frame, sticker_img, mouth_center, mouth_size)
            elif position == "left_ear":
                self.add_sticker_to_frame(frame, sticker_img, left_ear_center, ear_size)
            elif position == "right_ear":
                self.add_sticker_to_frame(frame, sticker_img, right_ear_center, ear_size)

    def add_sticker_to_frame(self, frame, sticker, center, size):
        try:
            # 调整贴纸大小
            if sticker.shape[1] != size or sticker.shape[0] != size:
                sticker = cv2.resize(sticker, (size, size))
            
            # 确定贴纸在帧中的位置
            x_offset = center[0] - size // 2
            y_offset = center[1] - size // 2
            
            # 检查贴纸是否超出帧的边界
            if (x_offset < 0 or y_offset < 0 or 
                x_offset + sticker.shape[1] > frame.shape[1] or 
                y_offset + sticker.shape[0] > frame.shape[0]):
                return
            
            # 如果贴纸有透明通道（4通道）
            if sticker.shape[2] == 4:
                # 分离RGB和透明通道
                rgb = sticker[:, :, 0:3]
                alpha = sticker[:, :, 3] / 255.0
                
                # 为透明通道创建广播数组
                alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
                
                # 为贴纸区域创建ROI
                roi = frame[y_offset:y_offset+sticker.shape[0], x_offset:x_offset+sticker.shape[1]]
                
                # 使用alpha通道混合贴纸和原始图像
                blended = (1.0 - alpha) * roi + alpha * rgb
                
                # 将混合结果放回原始图像
                frame[y_offset:y_offset+sticker.shape[0], x_offset:x_offset+sticker.shape[1]] = blended
            else:
                # 如果没有透明通道，直接叠加
                frame[y_offset:y_offset+sticker.shape[0], x_offset:x_offset+sticker.shape[1]] = sticker
        except Exception as e:
            print(f"添加贴纸到帧失败: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FaceSwapperGUI()
    window.show()
    sys.exit(app.exec_()) 