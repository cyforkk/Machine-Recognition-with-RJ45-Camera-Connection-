# 机器识别（连接rj45网络摄像头）

# Rj45网络摄像头连接与IP地址修改、开启RTSP协议

### 需要设备

- 一台具有poe接口（自带供电）的交换机
- 一台路由器



### 连接摄像头

> `声明我的摄像头ip是修改过的192.168.3.127`
>
> 正确第一次连接摄像头是在192.168.1.1这个网段

- 将光纤一头连接`摄像头`，一头连接交换机的`poe口`
- 将光纤一头连接交换机的`uplink口`，一头连接电脑的`网口`
- 将电脑连接的网口的ip改为与摄像头`同一网段`的IP地址，摄像头的IP地址为192.168.1.123，改为`192.168.1.x`![image-20250915100856233](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250915100856233.png)
- 去浏览器访问`192.168.1.123`，安装他要下载的`插件`
- ![image-20250915100330297](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250915100330297.png)
- 安装完毕，再次访问，还是进不去，打开这个`在Internet explorer模式下加载`（打开浏览器IE模式）![image-20250915100415655](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250915100415655.png)



- ![image-20250915100508858](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250915100508858.png)







- 进入登录页面输入账号`admin`和密码`123456`，正常显示即表示连接成功![image-20250915101021566](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250915101021566.png)



### 修改摄像头IP地址

- 右上角的配置按钮，进入以太网设置，将IP地址和网关地址还有DNS1修改问对应的IP地址（目的是为与路由器在同一个网段）![image-20250915101546403](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250915101546403.png)



- 修改电脑IP地址为与摄像头同一个网段，再次访问你刚才修改的摄像头IP地址，如果能进入即时修改成功。







### 打开RTSP协议（高实时性）

- 进入流媒体设置，启用RTSP认证

![image-20250915102225465](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250915102225465.png)



# 机器识别

### 下载失败就是开了梯子

### Anaconda

- **作用：包管理（集中，有序）和环境管理（版本切换）**
- **使用conda命令对虚拟环境创建、删除**
- **自带python解释器**

---

#### pip（python自带的包管理工具）与conda比较

> **依赖项检查**
>
> pip：不一定会展示所需其他依赖包
>
> 安装包时或许会直接忽略依赖项而安装，仅在结果中提示错误
>
> conda：列出所需其他依赖包
>
> 安装包时自动安装其依赖项
>
> 可以便捷地在包的不同版本中自由切换
>
> **环境管理**
>
> pip：维护多个环境难度较大
>
> conda：比较方便地在不同环境之间进行切换，环境管理较为简单
>
> 对系统自带Python的影响
>
> pip：在系统自带Python中包的更新/回退版本/卸载将影响其他程序
>
> conda：不会影响系统自带Python
>
> **适用语言**
>
> pip：仅适用于Python
>
> conda：适用于Python, R, Ruby, Lua, Scala, Java, JavaScript, C/C++, FORTRAN

---

#### 安装时把杀毒软件给关闭了，大概率成功安装

#### 配置环境变量

在用户环境变量（path）去配置你路径下的这三个对应的路径

![image-20250519134142311](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250519134142311.png)

#### 镜像源

##### conda

这一步非常重要！因为Anaconda的下载源默认在国外，如果不配置我们国内源的话，下载速度会慢到你怀疑人生的。而且很多时候会导致网络错误而下载失败。配置方法如下：
打开cmd，执行以下命令，将清华镜像配置添加到Anaconda中：

```cmd
# 移除所有自定义镜像源
conda config --remove-key channels

# 重新添加清华大学镜像源
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes
# 查看所有配置信息（channel是镜像源）
conda config --show

```

会生成一个文件在用户目录下文件名为`.condarc`



##### pip

```cmd
# 查看当前pip源
pip config list
# 修改镜像源
pip config set global.index-url 新源地址
pip config set global.index-url 
# 举例
https://pypi.tuna.tsinghua.edu.cn/simple
```



#### 命令

##### 测试是否安装成功

```cmd
conda --version
```

测试python版本

```
python --version
```



##### 进入/退出python解释器

```
#进入
python -v
#退出
quit
#清屏 
cls
```



##### 安装包（建议pip，pip没有的包就conda）

- **conda方式**

```python
# 在当前环境中安装包
conda install 包名称
# 指定版本号
conda install 包名称=version
# 在指定环境中安装包
conda install -n 环境名 包名称
```

- ##### pip方式

```
pip install 包名称 -i https://pypi.tuna.tsinghua.edu.cn/simple   #清华镜像
pip install 包名称 -i  https://pypi.douban.com/simple    #豆瓣镜像
```



##### 查看包（当前环境）

```
conda list
```



##### 更新包

先更新conda

```
conda updata conda
```

再更新第三方所用包

```
conda upgrade --all
```



##### 激活环境

```
conda activate 环境名
```

##### 查看当前存在的环境（最初环境名为base）

```
 conda info --envs
```

##### 创建其他python版本的虚拟环境（没有对应的版本时使用）

conda create -n 名字  python=版本号

```
conda create -n test python=3.8
```

会下载，下载完路径再anaconda3路径下的envs文件夹下

![image-20250519140534449](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250519140534449.png)

##### 切换虚拟环境

```
activate test
```

**注意，如果你之前用过conda activate xxx多次进入不同的环境操作之后，然后使用conda deactivate是返回上一层的环境。**

##### 退出环境

```
deactivate
```



##### 删除环境/包

**以上的-n均可用–name代替**

```
# ，可以删除指定环境（谨慎操作）
conda remove -n 环境名 --all -y
# 可以删除当前环境的包
conda remove 包名称
# 卸载指定环境中的包
conda remove -n 环境名 包名称
```



##### 其他命令

![image-20250519173452896](C:\Users\11390\AppData\Roaming\Typora\typora-user-images\image-20250519173452896.png)



### 项目

##### 1、新建conda环境

进入cmd，输入

```
conda create -n 虚拟环境名 python=3.8
```

没有成功就是没有安装anaconda软件，或者没有给软件配置环境变量



##### 2、激活环境

```
conda activate 环境名
```



# 从零开始配置目标检测项目依赖：稳定兼容的版本方案

在搭建目标检测或图像处理项目时，依赖版本冲突是最常见的 “拦路虎”。本文提供一套**从零开始的完整配置方案**，从虚拟环境创建到核心库安装，每个步骤都兼顾兼容性与功能性，即使是新手也能顺利复现，且全程保护你的私人信息（如路径、用户名等）。

### 没有Anaconda去看这个篇文章[Anaconda配置环境变量和镜像](https://blog.csdn.net/m0_60277481/article/details/151659807?fromshare=blogdetail&sharetype=blogdetail&sharerId=151659807&sharerefer=PC&sharesource=m0_60277481&sharefrom=from_link)

## 一、准备工作：创建独立虚拟环境

首先，为项目创建专属虚拟环境（避免污染全局环境），以 Anaconda 为例：

```bash
# 创建名为“bisai”的虚拟环境（Python 3.8兼容性最佳）
conda create -n bisai python=3.8 -y

# 激活环境（后续所有操作均在该环境中执行）
conda activate bisai
```

**为什么用 Python 3.8？**

它是兼容旧版 PyTorch（如 1.8.x）和新版工具（如 FastAPI 0.100.0）的 “黄金版本”，避免因 Python 版本过高导致的库兼容性问题。

## 二、核心依赖安装（按顺序执行）

按以下步骤安装，可最大程度减少版本冲突，每个命令都附带 “版本选择理由”：

### 1. 安装 PyTorch 与 GPU 支持（核心中的核心）

GPU比CPU图像识别更快（有则用）

选择自己想匹配的版本，我这个不适配所有版本

```bash
# 安装PyTorch 2.3.1（支持CUDA 12.1+，适配多数NVIDIA显卡）
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**版本理由**：

- 支持 GPU 加速（需 NVIDIA 显卡，如 RTX 3050），推理速度比 CPU 快 10 倍以上；

- 与后续安装的ultralytics（YOLO 库）完美兼容，避免新版 PyTorch 的 API 变更导致报错。

### 2. 安装 Web 框架与服务器

```bash
# 安装FastAPI和运行服务器
pip install fastapi==0.100.0 uvicorn==0.23.2
```

**作用与版本理由**：

- fastapi==0.100.0：轻量高性能的 API 框架，用于搭建 “接收图像→返回检测结果” 的接口；

- uvicorn==0.23.2：运行 FastAPI 的服务器，轻量且稳定，不与其他库冲突。

### 3. 安装数据处理与验证库

```bash
# 安装数据验证和基础处理工具
pip install pydantic==1.10.12 numpy==1.23.5 Pillow==9.5.0
```

**作用与版本理由**：

- pydantic==1.10.12：为 FastAPI 提供数据验证（确保输入的图像参数合法），兼容后续的 Gradio；

- numpy==1.23.5：处理图像数组（YOLO 和 OpenCV 的核心依赖），避免高版本（如 2.x）与旧库的编译冲突；

- Pillow==9.5.0：轻量图像处理（格式转换、保存等），与 numpy 1.23.5 完美适配。

### 4. 安装 YOLO 模型库与 OpenCV

```bash
# 安装YOLO官方库和计算机视觉工具
pip install ultralytics==8.3.185 opencv-python==4.7.0.72
```

**作用与版本理由**：

- ultralytics==8.3.185：YOLO 目标检测官方库，支持 YOLOv8/11 模型，与 PyTorch 2.3.1 兼容；

- opencv-python==[4.7.0.72](http://4.7.0.72)：复杂图像处理（如视频帧截取），与 numpy 1.23.5 无冲突。

### 5. 安装可视化调试工具（可选）

如果需要快速生成交互界面（上传图片→显示检测结果），安装 Gradio：

```bash
pip install gradio==3.41.0
```

**版本理由**：

Gradio 3.41.0 支持pydantic 1.x，与现有环境兼容；而 4.x 版本强制要求pydantic 2.x，会引发冲突。

## 三、验证安装：确保所有库正常工作

创建[verify.py](http://verify.py)文件，复制以下代码，运行验证：

```bash
# 验证核心库是否正常加载
import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI
from ultralytics import YOLO
import gradio as gr

# 检查PyTorch GPU支持
print(f"PyTorch版本: {torch.__version__}")  # 应输出2.3.1+cu121
print(f"GPU是否可用: {torch.cuda.is_available()}")  # 应输出True（若有NVIDIA显卡）

# 检查YOLO模型加载
try:
    model = YOLO("yolo11n.pt")  # 加载轻量模型
    print("YOLO模型加载成功")
except Exception as e:
    print(f"YOLO模型加载失败: {e}")

# 检查图像处理库
try:
    # 创建测试图像并保存
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.imwrite("test.jpg", img)
    Image.open("test.jpg")  # 用Pillow打开
    print("图像处理库工作正常")
except Exception as e:
    print(f"图像处理库错误: {e}")

print("所有验证通过！")
```

运行验证：

```bash
python verify.py
```

**成功标志**：所有检查均输出 “成功”，无报错信息。

## 四、避坑指南：新手常遇问题解决

1. **“GPU 是否可用” 输出 False？**

- - 检查显卡是否为 NVIDIA，且已安装对应驱动（如 RTX 3050 需驱动≥556.12）；

- - 重新安装 PyTorch 时确保带+cu121后缀（CPU 版无 GPU 支持）。

1. **安装时提示 “找不到版本”？**

- - 替换 pip 源为官方源（避免镜像源缓存旧版本）：

```bash
pip install 包名==版本号 --no-cache-dir --index-url https://pypi.org/simple
```

1. **Gradio 启动时报错？**

- - 确保pydantic版本是 1.10.12（而非 2.x），卸载冲突版本：

```bash
pip uninstall -y pydantic && pip install pydantic==1.10.12
```

## 五、总结：依赖配置的核心原则

1. **虚拟环境隔离**：用conda create创建独立环境，避免全局污染；

1. **版本严格匹配**：核心库（如 PyTorch 与 ultralytics、numpy 与 OpenCV）版本必须 “成对出现”，不能随意混搭；

1. **从基础到复杂**：先装底层库（PyTorch、numpy），再装上层工具（FastAPI、Gradio），减少依赖解析冲突；

1. **优先稳定版**：旧版本（如 numpy 1.23.5）虽然不是最新，但兼容性经过验证，适合生产环境。

按这套方案配置后，你将拥有一个稳定的目标检测项目环境，可直接用于开发 YOLO 模型推理、API 接口部署或可视化调试界面。