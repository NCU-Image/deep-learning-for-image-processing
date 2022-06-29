file=open('modelone.txt', 'r')
modelone= file.readlines()
print(modelone)
file=open('modeltwo.txt', 'r')
modeltwo= file.readlines()
print(modeltwo)
# file=open('modelmix.txt', 'r')
# modelmix= file.readlines()
# print(modelmix)
file=open('data.txt', 'r')
data= file.readlines()
print(data)
lenth=len(data)
print(lenth)
modelmix=[]
all=0
dui =0
cuo =0
for i in range(lenth):
    if  modelone[i] == '2\n'  and  modeltwo[i]  =='2\n':
        modelmix.insert(i,'2\n')
    else:
        modelmix.insert(i,'1\n')
for i in range(lenth):
    if modelone[i] == '2\n' and  modeltwo[i] == '2\n' :
        modelmix.append('2\n')
    else:
        modelmix.append('1\n')

print(modelmix)
print(len(modelmix))

for i in range(lenth):
    if modelmix[i] == data[i]:
        dui +=1
    else:
        cuo +=1



print("正确个数:", dui, "错误个数：", cuo)
accuracy = int(dui) / int(lenth)
print("测试总数:", lenth, "准确率：", accuracy)
