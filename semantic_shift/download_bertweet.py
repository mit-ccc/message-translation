from transformers import AutoModel, AutoTokenizer 

bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

bertweet.save_pretrained("./models/bertweet/")