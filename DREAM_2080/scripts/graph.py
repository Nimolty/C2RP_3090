from matplotlib import pyplot as plt
#画图
dta =  open('../../loss_data.txt')
dct = {
    'loss_1': [],
    'loss_2' : [],
    'loss_t' : [],
    'loss_r' : [],
    'loss_mc' : [],
    'loss_t1' : [],
    'loss_t2' : [],
    'loss':[]
}
x = []
n = 0
for line in dta:
    if line[0] == 'e':
        x.append(n)
        n += 1
    if line[0] != 'l':
        continue
    line = line.strip('\n')
    line = line.split(':')
    dct[line[0]].append(line[1])

mc = dct['loss_t']
mc = [float(i) for i in mc]
for i in range(len(mc)):
    if mc[i] > 2000:
        mc[i] = mc[i-1]

# 绘制图形
plt.plot(range(61), mc[0::10])
# plt.show()
plt.savefig('../../loss_t.png')
dta.close()

