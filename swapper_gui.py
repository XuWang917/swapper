import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QGridLayout, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

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
        self.selected_face_idx = -1
        self.current_source_face = None
        
        # 初始化摄像头
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 设置界面
        self.setup_ui()
        
        # 加载默认人脸
        self.load_default_faces()

    def setup_ui(self):
        # 主布局
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧：人脸库面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 人脸库标题
        title_label = QLabel("人脸库")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        left_layout.addWidget(title_label)
        
        # 人脸网格容器（可滚动）
        self.face_scroll = QScrollArea()
        self.face_scroll.setWidgetResizable(True)
        self.face_container = QWidget()
        self.faces_grid = QGridLayout(self.face_container)
        self.face_scroll.setWidget(self.face_container)
        left_layout.addWidget(self.face_scroll)
        
        # 底部按钮
        buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("加载更多")
        self.load_button.clicked.connect(self.load_more_faces)
        buttons_layout.addWidget(self.load_button)
        left_layout.addLayout(buttons_layout)
        
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
        
        # 操作按钮
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("开始换脸")
        self.start_button.clicked.connect(self.toggle_face_swap)
        self.start_button.setEnabled(False)
        control_layout.addWidget(self.start_button)
        right_layout.addLayout(control_layout)
        
        # 设置比例
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)
        
        self.setCentralWidget(central_widget)
    
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
                    self.add_face_to_library(filepath)
        else:
            print("faces文件夹为空")
            
        # 如果没有默认人脸，加载source_face.jpg
        if len(self.face_images) == 0 and os.path.exists("source_face.jpg"):
            print("尝试加载默认人脸: source_face.jpg")
            self.add_face_to_library("source_face.jpg")
    
    def add_face_to_library(self, image_path):
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
                return
                
            print(f"检测到人脸数量: {len(faces)}")
            
            # 获取第一个人脸
            face_feature = faces[0]
            
            # 存储图片和特征
            self.face_images.append(img)
            self.face_features.append(face_feature)
            
            # 更新界面
            self.update_face_grid()
            print(f"成功添加人脸: {image_path}")
            
        except Exception as e:
            print(f"添加人脸失败: {image_path}")
            print(f"错误详情: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_face_grid(self):
        # 清空网格
        while self.faces_grid.count():
            item = self.faces_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 重新填充网格
        for i, img in enumerate(self.face_images):
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
                label.setStyleSheet("padding: 5px; border: 2px solid blue;")
            
            # 添加到网格
            row, col = divmod(i, 2)
            self.faces_grid.addWidget(label, row, col)
    
    def select_face(self, idx):
        self.selected_face_idx = idx
        self.current_source_face = self.face_features[idx]
        self.update_face_grid()
        self.start_button.setEnabled(True)
    
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
        else:
            if self.selected_face_idx == -1:
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
                return
            
            print(f"成功打开摄像头")
            self.timer.start(30)  # 约30FPS
            self.start_button.setText("停止换脸")
    
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
            if target_faces:
                target_face = target_faces[0]
                display_frame = self.swapper.get(frame, target_face, self.current_source_face, paste_back=True)
            else:
                # 如果没检测到人脸，在帧上显示提示
                cv2.putText(display_frame, "未检测到人脸", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(f"换脸失败: {str(e)}")
        
        # 转换帧并显示
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        q_img = QImage(rgb_frame.data, w, h, w * ch, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # 缩放以适应标签
        pixmap = pixmap.scaled(self.preview_label.width(), self.preview_label.height(), 
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FaceSwapperGUI()
    window.show()
    sys.exit(app.exec_()) 