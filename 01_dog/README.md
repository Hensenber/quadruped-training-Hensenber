一、本任务要求：使狗趴在平坦地面上

拆分为四个检查点：
1.使狗能单独在mujoco中显示
2.使floor能在mujoco中单独显示
3.将floor与狗组合在同一个模型中
4.处理趴卧姿态 + actuator 力矩模式 + ctrl=0 + 仿真循环

二、配置虚拟环境
本次任务由于需要使用到mujoco库，我们尝试自己配置虚拟环境。
在01_dog根目录使用

python3 -m venv .venv

来创建虚拟环境

使用

source .venv/bin/activate               #输出类似(.venv) hensenber@ZHANG:~/robocon/training/01_dog$

来进行激活

使用

which python

来验证pyyhon是不是在这个虚拟环境里

最后安装对应的库即可（本题中是mujoco）

python -m pip install mujoco

安装完成后可以使用

python -c "import mujoco; print(mujoco.__version__)"

进行版本号验证



三、.gitigonre
本题中我们配置了虚拟环境，一般来说虚拟环境不需要被git管理，因此我们在项目根目录用nano增添.gitignore,在里面写入/01_dog/.venv/，来将其ignore掉。

需要注意的是，.gitignore 本身是要commit的



# Quadruped Dog MuJoCo Simulation

## 1. Project Overview

简单介绍：
- 本项目旨在建立一个平面，让给定的机器狗在平面上呈现稳定的卧姿
- 使用了MuJoCo软件进行仿真，使用Python语言书写simulation程序，使用UFDR文件保存机器人，实际读取时转换成了MJCF文件

## 2. Project Structure

写项目目录树，并解释：
- robot/       保存机器人的mesh和MJCF文件
- scene/       保存了平地场景floor的MJCF，以及最终场景（包含机器狗和scene）的MJCF
- simulate.py  模拟程序，用以启动python
- README.md    书写项目书
- .venv        虚拟环境，配置了MUJOCO库

## 3. Environment

例如：
- Ubuntu 22.04
- Python 3.10
- MuJoCo


