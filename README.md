Using natural language processing (NLP) technologies to develop medical chatbots makes the diagnosis of the patient more convenient and efficient, which is a typical application in healthcare AI. Because of its importance, lots of researches have come out. Recently, the neural generative models have shown their impressive ability as the core of chatbot, while it cannot scale well when directly applied to medical conversation due to the lack of medical-specific knowledge. To address the limitation, a scalable medical knowledge-assisted mechanism (MKA) is proposed in this paper. The mechanism is aimed at assisting general neural generative models to achieve better performance on the medical conversation task. The medical-specific knowledge graph is designed within the mechanism, which contains 6 types of medical-related information, including department, drug, check, symptom, disease, and food. Besides, the specific token concatenation policy is defined to effectively inject medical information into the input data. Evaluation of our method is carried out on two typical medical datasets, MedDG and MedDialog-CN. The evaluation results demonstrate that models combined with our mechanism outperform original methods in multiple automatic evaluation metrics. Besides, MKA-BERT-GPT achieves state-of-the-art performance.

Step 1: MKA Mechanism ----- Data Processing
It is based on the data not only for multiple rounds of dialogue data, but also for the related patient self-reports and the self-generated medical knowledge graphs. The knowledge graphs are generated based on the self-report (EHR) in each turns. Meanwhile, the medical key words in the dialogue data in each turns also make efforts on the graph genereation procedure. In the Data-Prepare folder there are 6 files:
1. dpt_text2kg.json: The map function between medical departments and departments in the knowledge graph.
2. key_body_symprelated.json: The map function between each body parts and its corresponding expressions.
3. kg_disease.txt: The disease entities in medical knowlegde graphs.
4. kg_symptom.txt: The symptom entities in medical knowlegde graphs.
5. Medical_Specific_Knowledge_Generator_new.py: Generating the medical knowledge sub-graph for each turn. 
6. data_gjy_seen.py: Combining the graphs and dialogue context as the input for the model.

Step 2: MKA Mechanism ----- Large Scale Model Application
2-1 Using Bert-GPT: Within bert-gpt folder, there are 4 files:
1. preprocess.py: Data preprocessing for Bert-GPT model.
2. bert_gpt_train_dpt.py: Training.
3. generate.py: Testing.
4. validate.py: Evaluation for Perplexity, BLEU, NIST, METEOR, Entropy, Dist.

2-2 Using Transformer: Within transformer folder, there are 3 files:
1. trans_preprocess.py: Data preprocessing for Transformer model.
2. trans_train.py: Training
3. trans_generate.py: Testing and evaluation.

If you use the repository, please cite the paper:
Ke Liang, Sifan Wu, Jiayi Gu, "MKA: A Scalable Medical Knowledge-Assisted Mechanism for Generative Models on Medical Conversation Tasks", Computational and Mathematical Methods in Medicine, vol. 2021, Article ID 5294627, 10 pages, 2021. https://doi.org/10.1155/2021/5294627
