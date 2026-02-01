# Towards Real-World Cross-View Geo-Localization: Robust Alignment via Geometric-Semantic Synergy
#### 💡This is a branch for testing function on the server, maintained by `XiaomoWen, NJUST-Automation`

> Here is the code repository for the remote sensing project of the 2026 College Students' Innovation and Enterprenuership Competition. The copyright belongs to the team led by Professor Bai Hongyang from the School of Energy and Engineering, NJUST. Addtionally, this paper is under the guidance of Teacher Guo Shuai.

### 📝Abstract 

$\qquad$ Real-world Cross-View Geo-Localization faces coupled degradations: geometric misalignment and environmental shifts, that disrupt standard ViTs trained on ideal data. 
		To bridge this gap, we present University-1652-C, a spectrally-validated benchmark for diagnostic stress-testing, and a Geometric-Robust Dual-Stream Framework. 
		Our approach enforces Dual-Invariance via Rotary Positional Embeddings (RoPE) and a Label-Guided Hard-aware Environmental Consistency (HEC) loss. 
		Crucially, adhering to a ``Train-Heavy, Deploy-Light'' paradigm, the auxiliary consistency stream is discarded during inference, ensuring zero additional computational cost. 
		Extensive experiments demonstrate that our method significantly outperforms state-of-the-art competitors in robust scenarios.

### 📝Notice (There are some points you must read before running code)

1.The dataset has not been upload, you should download it yourself through this website: https://drive.google.com/file/d/1iVnP4gjw-iHXa0KerZQ1IfIO0i1jADsR/view.

2.The official pre-trained model of Vision Transformer has not been upload,you should download it yourself through this website: https://drive.google.com/file/d/1-Rp-VAlUdb_dRbquyWhD6pdkhXYrwu-w/view, and copy it into the folder `main_project/pretrain_model` ( if there isn't a folder named this, create it by yourself and create a empty file named `__init__.py` in this folder.)

3.Remember to change the document path in script `train_test_local.sh`.(Some frequently-used hyper-parameters are also explicitly specified in this script)

4.After configuring your preferred parameters in `train_test_local.sh`, you can launch the script to perform training and automatically test using the generated weights with the following command: `bash ./main_project/train_test_local.sh`.

5.Branch `test_server` only saves our models, branch `main` may contain other SOTA models. You can learn them by `git checkout main`.
