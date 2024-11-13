from PIL import Image
img=Image.open('C:/Users/Taraksh Goyal/Desktop/coding/python/siketlearn/five.png')
data=list(img.getdata())
for i in range (len(data)):
    data[i]=255-data[i]
