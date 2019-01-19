# 车牌识别的思路

车牌识别的思路其实就是和验证码识别是一样的道理，在中国的车牌中一共有7位字符可以对这7个字符进行one-hot编码生成一个向量
一张车牌对应一个车牌标签即这7个字符生成的向量

# 使用google的colaboratory训练resnet34模型
train.ipynb是在colaboratory中运行的

# 在model.fit_generator()可以调参，我没有调参使用的默认的参数，调参数可以上keras查看模型调参函数的设置
#训练的时候每轮1000个样本共5轮，一个batch_size=32，所以一共有16W张图片

model.fit_generator(gen(), samples_per_epoch=1000,nb_epoch=5, 
                    nb_worker=1, pickle_safe=True, 
                    validation_data=gen(), nb_val_samples=1280)
                    
# test.py是使用训练好的模型进行测试
调用评价函数时：
跑完后的是每个字符的准确率不是整张图片的准确率

# 生成测试图片.py是可以生成供test.py测试的图片
