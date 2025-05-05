import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QGridLayout, QScrollArea)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

class FaceSwapperGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时换脸 (GPU版)")
        self.setMinimumSize(1000, 600)
        
        # 检测CUDA是否可用
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.providers = ["CUDAExecutionProvider"] if self.cuda_available else ["CPUExecutionProvider"]
            
        # 初始化模型
        self.app = FaceAnalysis(name="buffalo_l", providers=self.providers)
        self.app.prepare(ctx_id=0, det_size=(256, 256))
        
        # 初始化换脸模型
        model_path = "./models/inswapper_128.onnx"
        if not os.path.exists(model_path):
            self.swapper = get_model(model_path, download=True, providers=self.providers)
        else:
            self.swapper = get_model(model_path, download=False, providers=self.providers)
        
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
        
        # 预览标题和状态
        preview_layout = QHBoxLayout()
        preview_label = QLabel("实时预览")
        preview_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        preview_layout.addWidget(preview_label)
        
        # 显示GPU状态
        gpu_status = "GPU加速已启用" if self.cuda_available else "使用CPU模式"
        gpu_label = QLabel(gpu_status)
        gpu_label.setStyleSheet("color: green; font-weight: bold;" if self.cuda_available else "color: orange;")
        preview_layout.addWidget(gpu_label, alignment=Qt.AlignRight)
        
        right_layout.addLayout(preview_layout)
        
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
        
        # 添加分辨率选择
        self.res_button = QPushButton("高分辨率模式")
        self.res_button.setCheckable(True)
        self.res_button.clicked.connect(self.toggle_resolution)
        control_layout.addWidget(self.res_button)
        
        right_layout.addLayout(control_layout)
        
        # 设置比例
        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 2)
        
        self.setCentralWidget(central_widget)
        
        # 状态栏
        self.statusBar().showMessage("就绪")
    
    def toggle_resolution(self, checked):
        # 确定分辨率
        det_size = (640, 640) if checked else (256, 256)
        resolution_name = "高" if checked else "低"
        
        # 更新按钮文本
        self.res_button.setText("低分辨率模式" if checked else "高分辨率模式")
        self.statusBar().showMessage(f"正在切换到{resolution_name}分辨率模式...")
        
        # 重新加载分析模型
        self.app = FaceAnalysis(name="buffalo_l", providers=self.providers)
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.statusBar().showMessage(f"已切换到{resolution_name}分辨率模式")
        
        # 如果已加载人脸，重新处理它们
        if self.face_images:
            self.statusBar().showMessage("正在使用新分辨率重新分析人脸...")
            self._reload_faces()
    
    def _reload_faces(self):
        """使用当前模型重新分析所有人脸"""
        temp_images = self.face_images.copy()
        self.face_images = []
        self.face_features = []
        
        # 保存当前选择的索引
        old_selected_idx = self.selected_face_idx
        self.selected_face_idx = -1
        self.current_source_face = None
        
        # 重新处理每个人脸
        for i, img in enumerate(temp_images):
            try:
                faces = self.app.get(img)
                if faces:
                    self.face_images.append(img)
                    self.face_features.append(faces[0])
                    
                    # 还原之前的选择
                    if i == old_selected_idx and self.selected_face_idx == -1:
                        self.select_face(len(self.face_images) - 1)
            except Exception:
                pass
        
        # 更新UI
        self.update_face_grid()
        self.statusBar().showMessage(f"已重新分析 {len(self.face_images)} 个人脸")
        
        # 如果之前有选中的人脸但现在没有了，禁用开始按钮
        if old_selected_idx >= 0 and self.selected_face_idx == -1:
            self.start_button.setEnabled(False)
    
    def load_default_faces(self):
        # 加载默认目录中的人脸图片
        default_dir = "faces"
        if not os.path.exists(default_dir):
            os.makedirs(default_dir)
            self.statusBar().showMessage("创建了faces文件夹，请添加人脸图片")
            return
        
        # 加载faces文件夹中的图片
        files = os.listdir(default_dir)
        if files:
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepath = os.path.join(default_dir, filename)
                    self.add_face_to_library(filepath)
        
        # 如果没有默认人脸，加载source_face.jpg
        if len(self.face_images) == 0 and os.path.exists("source_face.jpg"):
            self.add_face_to_library("source_face.jpg")
            
        self.statusBar().showMessage(f"已加载 {len(self.face_images)} 个人脸")
    
    def add_face_to_library(self, image_path):
        try:
            # 处理中文路径问题
            if os.path.exists(image_path):
                # 使用numpy直接读取文件，避免cv2中文路径问题
                img = np.fromfile(image_path, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            else:
                self.statusBar().showMessage(f"文件不存在: {os.path.basename(image_path)}")
                return
                
            if img is None:
                self.statusBar().showMessage(f"无法读取图片: {os.path.basename(image_path)}")
                return
                
            # 分析人脸
            self.statusBar().showMessage(f"正在分析人脸: {os.path.basename(image_path)}")
            faces = self.app.get(img)
            
            if not faces:
                self.statusBar().showMessage(f"未检测到人脸: {os.path.basename(image_path)}")
                return
            
            # 获取第一个人脸
            face_feature = faces[0]
            
            # 存储图片和特征
            self.face_images.append(img)
            self.face_features.append(face_feature)
            
            # 更新界面
            self.update_face_grid()
            self.statusBar().showMessage(f"已加载人脸: {os.path.basename(image_path)}")
            
        except Exception as e:
            self.statusBar().showMessage(f"加载失败: {str(e)[:30]}")
    
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
            
            # GPU加速图像缩放
            if self.cuda_available:
                try:
                    gpu_img = cv2.cuda_GpuMat()
                    gpu_img.upload(img)
                    gpu_img_resized = cv2.cuda.resize(gpu_img, (thumb_w, thumb_h))
                    thumb = gpu_img_resized.download()
                except:
                    thumb = cv2.resize(img, (thumb_w, thumb_h))
            else:
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
            
            # 修复点击事件的lambda函数问题
            def create_click_handler(index):
                return lambda event: self.select_face(index)
            
            label.mousePressEvent = create_click_handler(i)
            
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
        self.statusBar().showMessage(f"已选择人脸 #{idx+1}")
    
    def load_more_faces(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        
        if file_dialog.exec_():
            filenames = file_dialog.selectedFiles()
            for filename in filenames:
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
            self.statusBar().showMessage("换脸已停止")
        else:
            if self.selected_face_idx == -1:
                return
            
            # 尝试不同的摄像头索引
            camera_indices = [0, 1, -1]
            for idx in camera_indices:
                self.cap = cv2.VideoCapture(idx)
                if self.cap.isOpened():
                    # 设置更高分辨率（如果摄像头支持）
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    break
            
            if not self.cap or not self.cap.isOpened():
                self.preview_label.setText("无法打开摄像头，请检查设备连接")
                self.statusBar().showMessage("错误: 无法打开摄像头")
                return
            
            self.timer.start(30)  # 约30FPS
            self.start_button.setText("停止换脸")
            self.statusBar().showMessage("换脸已开始")
    
    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # 水平翻转图像（镜像），使其更直观
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        
        # 进行换脸
        try:
            start_time = cv2.getTickCount()
            
            target_faces = self.app.get(frame)
            if target_faces:
                target_face = target_faces[0]
                display_frame = self.swapper.get(frame, target_face, self.current_source_face, paste_back=True)
                
                # 计算FPS
                end_time = cv2.getTickCount()
                processing_time = (end_time - start_time) / cv2.getTickFrequency()
                fps = 1.0 / processing_time
                
                # 在帧上显示FPS
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                self.statusBar().showMessage(f"FPS: {fps:.1f}")
            else:
                # 如果没检测到人脸，在帧上显示提示
                cv2.putText(display_frame, "未检测到人脸", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.statusBar().showMessage("未检测到人脸")
        except Exception as e:
            self.statusBar().showMessage(f"换脸失败: {str(e)[:30]}")
        
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