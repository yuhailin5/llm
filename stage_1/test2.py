'''
Author: YhL YhL00004_17@163.com
Date: 2026-04-03 10:56:35
LastEditors: YhL YhL00004_17@163.com
LastEditTime: 2026-04-10 15:22:32
FilePath: \py\stage_1\test2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
num = 3.1415926

#print(f"{num:.2f}")

# 生成器yeild
def count(n):
    while n > 0:
        yield n
        n -= 1
for i in count(5):
    print(i)


a = [1, 2, 3, 4, 5]
a.extend([6, 7, 8])
print(a)