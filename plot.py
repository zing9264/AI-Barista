import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot(array):
    df=pd.DataFrame(array)
    a_f=a_a=0
    b_f=b_a=0
    c_f=c_a=0
    d_f=d_a=0
    for i in range(array.shape[1]):
        if array[0][i]>=400 and array[0][i]<600:
            a_f+=1
            a_a+=array[1][i]
        elif array[0][i]>=600 and array[0][i]<800:
            b_f+=1
            b_a+=array[1][i]
        elif array[0][i]>=800 and array[0][i]<1100:
            c_f+=1
            c_a+=array[1][i]
        elif array[0][i]>1100 and array[0][i]<1500:
            d_f+=1
            d_a+=array[1][i]
    #    elif array[0][i]<400:
    #        e+=1

    print(df)
    print(array.shape)
    print(array[0][0])
    print(array[0][1])
    #a/=2.515
    #b/=1.85
    c_a/=1.36
    print('feret: '+str(a_f)+' , '+str(b_f)+' , '+str(c_f)+' , '+str(d_f))
    print('area: '+str(a_a)+' , '+str(b_a)+' , '+str(c_a)+' , '+str(d_a))
    #print(a+b+c+d+e)
    total_a=a_a+b_a+c_a+d_a
    total_a/=100
    labels='400~600','600~800','800~1100','>1100'
    size=[a_a,b_a,c_a,d_a]
    explode = (0.015, 0.015, 0.015, 0.015)

    plt.figure()
    plt.pie(size,                           # 數值
            labels = labels,                # 標籤
            autopct = "%1.1f%%",            # 將數值百分比並留到小數點一位
            pctdistance = 0.6,              # 數字距圓心的距離
            textprops = {"fontsize" : 10},  # 文字大小
            explode=explode,
            )
    plt.axis('equal')
    plt.title("particle distribution", {"fontsize" : 20})
    plt.legend(loc = "center right")
    plt.savefig("/static/images/pie.png")
    propotion=np.array([a_a/total_a,b_a/total_a,c_a/total_a,d_a/total_a])
    #print(propotion)
    plt.figure()
    #plt.style.use('ggplot')
    plt.title("particle distribution", {"fontsize" : 20})
    plt.xlabel("(μm)")
    plt.ylabel("(%)")
    plt.bar(labels, height=propotion , width=0.5, bottom=0)
    plt.savefig("/static/images/bar.png")
    return