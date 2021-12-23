Data-prepare文件夹：数据预处理，涉及数据包括知识图谱、患者自述报告以及多轮对话数据。利用患者自述报告中的信息获取知识图谱相关子图，根据对话信息中的关键词与知识图谱子图进行实体链接，检索出KG中的相关信息与对话信息相结合。
1、dpt_text2kg.json
 好大夫科室与知识图谱中科室映射表
2、key_body_symprelated.json
 身体部位与该部位常见表述映射表
3、kg_disease.txt
 知识图谱中的疾病实体
4、kg_symptom.txt
 知识图谱中的症状实体
5、Medical_Specific_Knowledge_Generator_new.py
 根据自述报告以及多轮对话数据从医学知识图谱中获取相关子图信息，存成json文件
6、data_gjy_seen.py
 将Medical_Specific_Knowledge_Generator_new.py生成的json文件，调整成模型所需的输入的形式。
 


bert-gpt文件夹 ：bert-gpt模型相关代码
1、preprocess.py
 数据处理模块，将经过data-prepare处理后的输入数据，向量化成模型需要的input。
1、bert_gpt_train_dpt.py 
 bert-gpt训练代码，输入是经过preprocess.py处理后的数据，训练后得到bert-gpt模型。
2、generate.py
 预测模块，输入为经过preprocess.py处理后的test数据，利用训练后保存的bert-gpt模型进行预测推理，生成txt预测结果文件。
3、validate.py
 模型评估模块，评估模型指标


transformer文件夹 ：transformer模型相关代码
1、trans_preprocess.py
 数据处理模块，将经过data-prepare处理后的结合KG的多轮对话数据，转成模型需要的input格式
1、trans_train.py 
 transformer训练代码，输入是经过preprocess.py处理后的数据，训练后得到transformer模型。
 2、trans_generate.py
 包括预测和评估两部分，输入为经过preprocess.py处理后的test数据，利用训练后保存的bert-gpt模型进行预测推理，生成txt预测结果文件，并评估模型指标
