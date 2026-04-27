'''
Author: YhL YhL00004_17@163.com
Date: 2026-04-02 11:21:28
LastEditors: YhL YhL00004_17@163.com
LastEditTime: 2026-04-03 10:56:45
FilePath: \py\stage_1\test1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import keyword
print(keyword.kwlist)
# 使用r转义将被忽略
s = r"1234567890"
a = '12'
b = s+a
c = a*3
d = a[0:2]
print(s)
# 实现按 enter 键后退出
input("\n\n按下 enter 键后退出。")

# 不换行输出
print(s,end="")