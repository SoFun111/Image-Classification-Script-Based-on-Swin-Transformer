一款基于 Swin Transformer V2 的图形化图片分类工具，支持自定义训练参数、自动保存配置、实时进度显示，专为特定格式文件夹（SDPicture_xxx）的图片分类设计，自动跳过含 shelf 关键词的文件夹，兼容 CPU/GPU 运行。
✨ 核心特性
- 📊 图形化界面（GUI）：无需命令行，点击即可操作，支持路径选择、模式切换。
- ⚙️ 自定义训练参数：可设置训练轮次（Epochs）、批次大小（Batch Size）、学习率（LR）。
- 📁 自动保存配置：所有路径、参数、模式设置自动保存到 config.json，下次打开自动恢复。
- 📈 实时反馈：内置进度条和日志框，训练/分类过程实时可见，报错自动提示。
- 🖼️ 智能图片处理：自动遍历 SDPicture_xxx 文件夹下所有子文件夹，跳过含 shelf 关键词的文件夹，支持 JPG/JPEG/PNG/BMP 格式。
- 🖥️ 双运行模式：支持训练模式（训练模型）和分类模式（批量分类图片）。
- 💻 多硬件兼容：自动检测 CUDA（GPU），无 NVIDIA 显卡时自动切换为 CPU 运行。
📋 功能说明
训练模式
基于 Swin Transformer V2 Tiny 模型，使用 ImageFolder 格式的数据集进行训练，自动划分训练集/验证集（9:1），支持早停（连续5轮无精度提升停止训练），最优模型保存为 swin_best_model.pth。
分类模式
加载训练好的模型，批量分类 SDPicture_xxx 文件夹下的所有图片，按分类结果和文件夹ID（SDPicture_xxx 中的ID）组织输出路径，自动跳过 shelf 相关文件夹。
📥 环境依赖
需安装 Python 3.8+，推荐使用虚拟环境（venv）安装依赖：
# 安装依赖
pip install -r requirements.txt
requirements.txt
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.5.0
tqdm>=4.65.0
numpy>=1.24.0
tkinter  # Python 自带，无需额外安装
pyinstaller>=5.13.0  # 用于打包EXE
🚀 使用步骤
1. 准备工作
- 训练模式：准备 ImageFolder 格式的训练集（文件夹名即为类别名，放入 train_data 文件夹或自定义训练路径）。
- 分类模式：确保已训练模型（swin_best_model.pth 与程序同目录），待分类图片放在 SDPicture_xxx 格式的文件夹下。
2. 运行程序
# 直接运行Python脚本
python main.py
3. 界面操作
1. 路径设置：点击「选择」按钮，分别设置训练路径（仅训练模式）、待分类路径（仅分类模式）、输出路径（仅分类模式）。
2. 训练参数：设置训练轮次、Batch Size、学习率（仅训练模式，默认值已优化，可直接使用）。
3. 模式切换：选择「训练模式」或「分类模式」。
4. 开始运行：点击「开始运行」，进度条和日志框将实时显示过程，完成后弹出提示。
📦 打包成EXE（可选）
如需打包为可执行文件，方便分发使用，执行以下命令：
# 单文件打包（无命令行窗口）
pyinstaller -F -w --onefile --strip --compress-args=3 main.py

# 非单文件打包（体积更小、启动更快）
pyinstaller -w --noconsole --strip main.py
打包完成后，EXE文件位于 dist 文件夹下，与 config.json、swin_best_model.pth 放在同一目录即可正常运行。
⚠️ 注意事项
- 警告说明：打包时出现 WARNING: Library not found: avdevice-58.dll 可忽略，该依赖属于 torchaudio（音频处理），本工具无需使用，不影响任何功能。
- GPU 占用：分类模式为单张图片推理，GPU 占用较低（5%-15%）是正常现象，只要显存占用>300MB，即说明正在使用 GPU 加速。
- 模型文件：训练完成后，模型将保存为 swin_best_model.pth，分类模式必须确保该文件存在。
- 配置文件：config.json 自动保存所有设置，删除该文件将恢复默认配置。
- 文件夹规则：仅处理 SDPicture_xxx 格式的文件夹，自动跳过名称含 shelf 的文件夹及子文件夹。
🔧 常见问题
Q1: 运行时提示“未找到模型文件，请先训练！”
A: 分类模式需要先运行训练模式，生成 swin_best_model.pth 模型文件，确保该文件与程序同目录。
Q2: 训练/分类时进度条不动、日志无输出？
A: 检查路径是否正确（确保是文件夹路径，而非文件路径），训练集是否为 ImageFolder 格式，待分类路径下是否有 SDPicture_xxx 文件夹及图片。
Q3: 打包后的EXE体积很大？
A: 正常现象，因打包时会包含 Python 解释器、PyTorch、Swin 模型等所有依赖，使用上述瘦身打包命令可将体积压缩至 800MB-1GB。
Q4: 如何确认是否使用了 GPU？
A: 打开任务管理器 → 性能 → GPU，查看「专用 GPU 显存」，若占用>300MB，说明已使用 GPU；无 NVIDIA 显卡则自动切换为 CPU 运行。
📄 文件说明
- main.py：程序主文件（含GUI、训练、分类逻辑）。
- config.json：自动生成的配置文件，保存路径、参数、模式设置。
- swin_best_model.pth：训练完成后生成的最优模型文件。
- requirements.txt：环境依赖清单。
