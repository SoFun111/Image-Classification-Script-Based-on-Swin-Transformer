import os
import re
import glob
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from shutil import copy2
from tqdm import tqdm

# ====================== 配置 ======================
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "train_dir": "./train_data",
    "source_dir": r"D:\download\tubephoto",
    "save_dir": r"D:\download\tubephoto",
    "img_size": 320,
    "batch_size": 2,
    "lr": 0.00005,
    "val_split": 0.1,
    "patience": 5,
    "epochs": 10
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("run.log", encoding="utf-8"), logging.StreamHandler()]
)
logger = logging.getLogger()

def pad_to_square(img):
    w, h = img.size
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    new_img.paste(img, ((max_dim - w)//2, (max_dim - h)//2))
    return new_img

val_transform = transforms.Compose([
    pad_to_square,
    transforms.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def create_swin_model(num_classes):
    model = models.swin_v2_t(pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
    return model.to(CONFIG["device"])

# ====================== 训练部分（不动） ======================
def train():
    logger.info("===== 训练开始 =====")
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(CONFIG["train_dir"], transform=val_transform)
    val_len = int(len(dataset)*CONFIG["val_split"])
    train_len = len(dataset)-val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = create_swin_model(len(dataset.classes))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    best_acc = 0
    early_stop = 0
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        for imgs,labels in tqdm(train_loader):
            imgs,labels = imgs.to(CONFIG["device"]), labels.to(CONFIG["device"])
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct=0
        total=0
        with torch.no_grad():
            for imgs,labels in val_loader:
                imgs,labels = imgs.to(CONFIG["device"]), labels.to(CONFIG["device"])
                correct += (model(imgs).argmax(1)==labels).sum().item()
                total += labels.size(0)
        acc = correct/total
        logger.info(f"Epoch {epoch+1} 精度: {acc:.4f}")

        if acc>=best_acc:
            best_acc=acc
            torch.save({"model":model.state_dict(),"classes":dataset.classes},"swin_best_model.pth")
            logger.info(f"✅ 最优模型保存 {best_acc:.4f}")
            early_stop=0
        else:
            early_stop+=1
            if early_stop>=CONFIG["patience"]:
                logger.info("⚠️ 早停")
                break
        scheduler.step(acc)


# ====================== 修复：遍历 SDPicture_xxx 下所有子文件夹 + 跳过 shelf ======================
def classify():
    logger.info("===== 开始分类 =====")
    ckpt = torch.load("swin_best_model.pth", map_location=CONFIG["device"])
    model = create_swin_model(len(ckpt["classes"])).to(CONFIG["device"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    tasks = []
    pattern = r"SDPicture_(\d+)_\d+"

    # 遍历第一层文件夹
    for folder_name in os.listdir(CONFIG["source_dir"]):
        folder_path = os.path.join(CONFIG["source_dir"], folder_name)
        if not os.path.isdir(folder_path):
            continue

        # 只处理 SDPicture_xxx
        if not re.match(pattern, folder_name):
            continue

        # 🔥 修复：跳过名称含 shelf 的文件夹
        if "shelf" in folder_name.lower():
            continue

        # 提取ID
        fid = re.match(pattern, folder_name).group(1)

        # 遍历该文件夹下 所有子文件夹 的图片
        for root, dirs, files in os.walk(folder_path):
            # 🔥 修复：跳过子文件夹里的 shelf
            if "shelf" in root.lower():
                continue

            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(root, f)
                    tasks.append((img_path, fid))

    logger.info(f"✅ 待分类图片总数: {len(tasks)}")

    for img_path, fid in tqdm(tasks, desc="分类中"):
        try:
            img = val_transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(CONFIG["device"])
            with torch.no_grad():
                pred_cls = ckpt["classes"][model(img).argmax().item()]

            target_dir = os.path.join(CONFIG["save_dir"], fid, pred_cls)
            os.makedirs(target_dir, exist_ok=True)
            copy2(img_path, target_dir)
        except Exception as e:
            logger.warning(f"失败: {img_path}")

    logger.info("===== 全部分类完成 =====")

if __name__ == "__main__":
    train()
    # classify()
