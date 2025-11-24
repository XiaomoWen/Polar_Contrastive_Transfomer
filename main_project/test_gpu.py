import torch
print(torch.__version__)          # 应该输出 PyTorch 版本号
print(torch.cuda.is_available())  # 必须输出 True
print(torch.cuda.device_count()) # 应该输出 1 或更大，表示检测到的显卡数量
print(torch.cuda.get_device_name(0)) # 应该输出显卡型号
exit()
