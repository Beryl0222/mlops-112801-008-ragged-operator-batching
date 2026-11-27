# 分段批化交换

模型副本、样本位置与单样本扩展函数在同一份张量交换资料中登记。

数据为合成记录，不依赖设备或在线服务。字段和单位见《交换约定》。
数据文件为 `fixtures/batched_axes.json`。当前可运行的检查核对这些输入及 NumPy 环境。

使用 Python 3.11。在未预装依赖的机器上先运行 `python -m pip install -r requirements.txt`。
在项目根目录执行 `python -m unittest discover -s tests -v`。
