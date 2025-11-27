import torch

def check_env():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"GPU Count: {device_count}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
        # 测试 Tensor 运算
        x = torch.tensor([1.0]).cuda()
        print("CUDA Tensor Test: Passed")
        
        # 关键：检查 NCCL (DDP 通信后端) 是否可用
        # 正常情况下 torch.distributed 需要初始化才能看，
        # 但我们可以简单检查是否有 built-in NCCL
        print(f"NCCL Available: {torch.distributed.is_nccl_available()}")
        
        # 检查 L40 的算力匹配 (Ada 应该是 8.9)
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: {cap[0]}.{cap[1]}")
        
        if cap[0] < 8:
            print("警告: 你的 PyTorch 版本可能没正确识别 Ada 架构，请检查 CUDA 版本。")
        else:
            print("完美: L40 架构识别正确！")

if __name__ == "__main__":
    check_env()