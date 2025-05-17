import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from torch.utils.data import DataLoader
from model import CSRNet
from dataset import CrowdDataset

import torch
# 그 뒤에 model import 등

def evaluate_model(model_path, image_dir, density_dir, batch_size=1):
    # 모델 로드
    model = CSRNet().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 검증 데이터셋
    dataset = CrowdDataset(image_dir, density_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    mae = 0.0
    mse = 0.0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.cuda()
            targets = targets.cuda()

            outputs = model(images)
            pred_count = outputs.sum().item()
            true_count = targets.sum().item()

            mae += abs(pred_count - true_count)
            mse += (pred_count - true_count) ** 2

    mae /= len(dataloader)
    rmse = (mse / len(dataloader)) ** 0.5

    print(f"\n✅ Evaluation Results for {os.path.basename(model_path)}")
    print(f"MAE  = {mae:.2f}")
    print(f"RMSE = {rmse:.2f}")

# 예시 실행 (Part_B validation)
if __name__ == "__main__":
    for i in range(1, 11):
        evaluate_model(
            model_path=f"csrnet_epoch{i}.pth",
            image_dir=r"archive\ShanghaiTech\part_B\test_data\images",
            density_dir=r"archive\ShanghaiTech\part_B\test_data\density-map"
        )
