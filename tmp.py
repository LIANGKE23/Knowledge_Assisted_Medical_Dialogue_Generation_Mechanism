import json

with open("train_data_new.json", 'r', encoding='utf-8') as file:
    data_train = json.load(file)
with open("validate_data_new.json", 'r', encoding='utf-8') as file:
    data_validate = json.load(file)
with open("test_data_new.json", 'r', encoding='utf-8') as file:
    data_test = json.load(file)

a=[0,0,0,0,0]
for i in data_train:
    c=len(i)
    if c==3:
        a[0]+=1
    elif c==5:
        a[1]+=1
    elif c==7:
        a[2]+=1
    elif c==9:
        a[3]+=1
    elif c==11:
        a[4]+=1
    else:
        print("错误")

print("训练集总数据量："+str(len(data_train)))
print("训练集轮数分布："+",".join([str(i) for i in a]))

a=[0,0,0,0,0]
for i in data_validate:
    c=len(i)
    if c==3:
        a[0]+=1
    elif c==5:
        a[1]+=1
    elif c==7:
        a[2]+=1
    elif c==9:
        a[3]+=1
    elif c==11:
        a[4]+=1
    else:
        print("错误")
print("验证集总数据量："+str(len(data_validate)))
print("验证集轮数分布："+",".join([str(i) for i in a]))

a=[0,0,0,0,0]
for i in data_test:
    c=len(i)
    if c==3:
        a[0]+=1
    elif c==5:
        a[1]+=1
    elif c==7:
        a[2]+=1
    elif c==9:
        a[3]+=1
    elif c==11:
        a[4]+=1
    else:
        print("错误")
print("测试集总数据量："+str(len(data_test)))
print("测试集轮数分布："+",".join([str(i) for i in a]))
# data_train_ = []
# for i in data_train:
#     a = i[0].split(",")[1]+","
#     j = ["病人：" + a + i[1][3:]] + i[2:]
#     data_train_.append(j)
# data_validate_ = []
# for i in data_validate:
#     a = i[0].split(",")[1]+","
#     j = ["病人：" + a + i[1][3:]] + i[2:]
#     data_validate_.append(j)
# data_test_ = []
# for i in data_test:
#     a = i[0].split(",")[1]
#     j = ["病人：" + a + i[1][3:]] + i[2:]
#     data_test_.append(j)
#
# with open("train_data_origin.json", 'w', encoding='utf-8') as file:
#     file.write(json.dumps(data_train_, indent=2, ensure_ascii=False))
# with open("validate_data_origin.json", 'w', encoding='utf-8') as file:
#     file.write(json.dumps(data_validate_, indent=2, ensure_ascii=False))
# with open("test_data_origin.json", 'w', encoding='utf-8') as file:
#     file.write(json.dumps(data_test_, indent=2, ensure_ascii=False))