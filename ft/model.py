import torch.nn as nn
import torch
from transformers import BertModel,RobertaModel, RobertaConfig,RobertaForMaskedLM


class BertClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        # self.bert = BertModel.from_pretrained(model_name)
        self.bert = RobertaModel.from_pretrained('../data/pretrain')
        # config = RobertaConfig.from_pretrained("weights/config.json")
        # # 加载模型
        #self.bert = torch.load("weights/bert-base-uncased_20250103_225634.pt")
        #self.bert = RobertaModel.from_pretrained('weights/bert-base-uncased_20250103_233333')
        #self.bert.config
        # # 加载微调后的权重
        # self.bert.load_state_dict(torch.load("weights/bert-base-uncased_20241224_144142.pt"))
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        #self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
    
    def save_pretrained(self, save_directory):
        """
        保存整个模型（包括 RoBERTa、dropout 和 classifier）到指定目录
        :param save_directory: 保存路径
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # 保存整个模型（包括 RoBERTa、dropout 和 classifier）
        self.bert.save_pretrained(save_directory)  # 保存 RoBERTa 模型和配置
        torch.save({
            "dropout_state_dict": self.dropout.state_dict(),
            "classifier_state_dict": self.classifier.state_dict()
        }, os.path.join(save_directory, "custom_layers.bin"))

        print(f"Model saved to {save_directory}")
        
    @classmethod
    def from_pretrained(cls, model_name, num_classes, save_directory):
        """
        从指定目录加载模型
        :param model_name: 预训练模型的名称或路径
        :param num_classes: 分类任务的类别数
        :param save_directory: 保存模型的路径
        :return: 加载的模型
        """
        import os
        # 加载 RoBERTa 模型
        model = cls(model_name, num_classes)
        model.bert = RobertaModel.from_pretrained(save_directory)

        print("jiazai")
        # 加载自定义层（dropout 和 classifier）
        custom_layers_path = os.path.join(save_directory, "custom_layers.bin")
        if os.path.exists(custom_layers_path):
            custom_layers = torch.load(custom_layers_path)
            model.dropout.load_state_dict(custom_layers["dropout_state_dict"])
            model.classifier.load_state_dict(custom_layers["classifier_state_dict"])
        else:
            print("Custom layers not found. Initializing new layers.")

        return model
