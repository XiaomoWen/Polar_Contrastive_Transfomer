# Polar Contrastive Transformer Network (PCT-Net)
### 💡A new branch for testing function on the server. 
**This branch is maintained by: `XiaomoWen, NJUST-Automation`** 

> This is the code repository for the remote sensing project of the 2026 College Students' Innovation and Enterprenuership Competition. The copyright belongs to the team led by Professor Bai Hongyang from the School of Energy and Engineering, NJUST.

### 📝Notice (There are some points you must read before running code)

1.The dataset "University-1652" has not been upload, you should download it yourself through this website: https://drive.google.com/file/d/1iVnP4gjw-iHXa0KerZQ1IfIO0i1jADsR/view.

2.The official pre-trained model of Vision Transformer(Small) has not been upload,you should download it yourself through this website: https://drive.google.com/file/d/1-Rp-VAlUdb_dRbquyWhD6pdkhXYrwu-w/view, and copy it into the folder `main_project/pretrain_model` ( if there isn't a folder named this, create it by yourself and create a empty file named `__init__.py` in this folder.)

3.Remember to change the document path in script `train_test_local.sh`.(Some frequently-used hyper-parameters are also explicitly specified in this script)

4.After configuring your preferred parameters in `train_test_local.sh`, you can launch the script to perform training and automatically test using the generated weights with the following command: `bash ./main_project/train_test_local.sh`.

5.Branch `test_server` only saves our models, branch `main` may contain other SOTA models. You can learn them by `git checkout main`.