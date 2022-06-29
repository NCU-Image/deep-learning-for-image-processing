import os 


with open('./train_3.txt','w') as f:
    after_generate = os.listdir(".\image/train")
    for image in after_generate:
        if image.split("_")[0]=='Chinese':
            f.write(image + ";" + "0" + "\n")
        elif image.split("_")[0]=='English':
            f.write(image + ";" + "1" + "\n")
        elif image.split("_")[0]=='German':
            f.write(image + ";" + "2" + "\n")
        else:f.write(image + ";" + "3" + "\n")

