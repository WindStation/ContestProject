from paddle.nn import functional as F
import paddle

print(paddle.device.get_device())
print("hello world")

paddle.device.set_device("cpu")
print("test_pull")
