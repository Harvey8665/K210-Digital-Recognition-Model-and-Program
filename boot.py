import sensor,image,lcd,time
import KPU as kpu
from machine import UART
from fpioa_manager import fm
lcd.init(freq=15000000)
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_hmirror(0)
sensor.run(1)
task = kpu.load("/sd/yolov3.kmodel")     # 此处修改为模型的名称，模型的文件类型为kmodel
f=open("anchors.txt","r")                # txt文件名需要用训练好的模型文件夹内的文件名替换
anchor_txt=f.read()
L=[]
for i in anchor_txt.split(","):
    L.append(float(i))
anchor=tuple(L)
f.close()
f=open("lable.txt","r")                  # txt文件名需要用训练好的模型文件夹内的文件名替换
lable_txt=f.read()
lable = lable_txt.split(",")
f.close()
fm.register(10, fm.fpioa.UART1_TX, force=True)
fm.register(11, fm.fpioa.UART1_RX, force=True)
uart_A = UART(UART.UART1, 115200, 8, 1, 0, timeout=1000, read_buf_len=4096)
anchor = (0.4192, 0.3702, 0.5744, 1.6689, 0.6932, 0.6054, 1.0054, 0.9615, 2.1672, 1.6683) #此处修改为anchors.txt中的内容
sensor.set_windowing((224, 224))
a = kpu.init_yolo2(task, 0.5, 0.3, 5, anchor)
classes=["9","1","4","2","3","8","5","6","7" ]  # 此处修改为lable.txt里面的内容
while(True):
     img = sensor.snapshot()
     code = kpu.run_yolo2(task, img)
     if code:
         for i in code:
             a=img.draw_rectangle(i.rect())
             a = lcd.display(img)
             list1=list(i.rect())
             b=(list1[0]+list1[2])/2
             c=(list1[1]+list1[3])/2
             #print("物体是：",classes[i.classid()])
             #print("概率为：",100.00*i.value())
             #print("坐标为：",b,c)
             for i in code:
                 lcd.draw_string(i.x(), i.y(), classes[i.classid()], lcd.RED, lcd.WHITE)
                 lcd.draw_string(i.x(), i.y()+12, '%f'%i.value(), lcd.RED, lcd.WHITE)
                 lcd.draw_string(50, 200,str(b), lcd.RED, lcd.WHITE)
                 lcd.draw_string(110, 200,str(c), lcd.RED, lcd.WHITE)
                 uart_A.write(classes[i.classid()]+'\r\n')
                 #uart_A.write(str(b)+'\r\n')
                 #uart_A.write(str(c)+'\r\n')
     else:
         a = lcd.display(img)
uart_A.deinit()
del uart_A
a = kpu.deinit(task)
