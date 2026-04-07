import json,os
from langchain_core.messages import message_to_dict, messages_from_dict,BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory

def get_history(session_id):
    return FileChatMessageHistory(session_id,"./chat_history")

class FileChatMessageHistory(BaseChatMessageHistory):
    
    def __init__(self,session_id,storage_path):
        self.session_id = session_id        #会话id
        self.storage_path = storage_path    #不同会话id存储文件夹路径
        
        self.file_path = os.path.join(self.storage_path, self.session_id)
        
        #确保文件夹路径存在
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
    def add_messages(self, messages) -> None:
        #Message : Sequence序列 类似list，tuple
        all_messages = list(self.messages) #已有的消息列表
        all_messages.extend(messages)
            
        new_message = [message_to_dict(message) for message in all_messages]
        # 将数据写入文件
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump(new_message, f)
    
    # 获取消息
    @property
    def messages(self) -> list[BaseMessage]: #通过property方法装饰器将messages方法变为成员属性
        # 当前文件内: list[字典]
        try:
            with open(self.file_path,"r",encoding="utf-8") as f:
                messages_data = json.load(f) #list[字典]
                return messages_from_dict(messages_data)
        except FileNotFoundError:
            return []
        
    def clear(self):
        with open(self.file_path,"w",encoding="utf-8") as f:
            json.dump([],f)
