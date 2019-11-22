# coding:utf-8
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from datetime import timedelta
from robotclient import start_Robot
from image_module import predict

#全域變數溝通
brewflag=0 #設定是否可沖煮
coffee_weight=0 #儲存咖啡重
#圖片格式允許
ALLOWED_EXTENSIONS = set(['png','jpeg' ,'jpg', 'JPG', 'PNG', 'bmp'])

def answer_normalize(arr=[]):
    total=0
    for i in range(len(arr)):
        print(arr,len(arr),i)
        total+=arr[i]
    ans=[]

    for i in range(len(arr)):
        print(arr,len(arr),i)
        if(total==0):
            ans.append(0)
        else:
            ans.append((arr[i]/total)*100)
    return ans

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
app = Flask(__name__)
#
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/', methods=['GET'])  # 
def index():
    global brewflag #設定是否可沖煮
    brewflag=0
    coffee_weight=0
    return render_template('index.html')
 
# @app.route('/upload', methods=['POST', 'GET'])
@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    global brewflag  #設定是否可沖煮

    if request.method == 'POST':
        i=0
        for file in request.files.getlist("file"):
            print("file " ,file,type(file),file.filename)

            if not(file and allowed_file(file.filename)):
                    # Make the filename safe, remove unsupported chars
                return jsonify({"error": 1001, "msg": "請檢查圖檔類型，僅限png、PNG、jpg、JPG、bmp"})
            basepath = os.path.dirname(__file__)  #當前目錄
            upload_path = os.path.join(basepath, 'static/images', secure_filename(file.filename))  
                # upload_path = os.path.join(basepath, 'static/images','test.jpg')  
            file.save(upload_path)
                # 使用Opencv轉換格式
            img = cv2.imread(upload_path)
            newname= "Demo_input.jpg"
            print(newname+"\n")
            cv2.imwrite(os.path.join(basepath, 'static/images/raw_photo',newname ), img)
            brewflag=1
            i = i+1
            break

        get_answer=[]
        for i in range(0,5):
            _="weight_"+str(i)
            print(request.form.get(_))
            j=int(request.form.get(_))
            get_answer.append(j)
        
        nor_ans=answer_normalize(get_answer)
        
        predict_answer=predict(correct_answer=nor_ans)
        #correct_answer = get_answer
        #predict_answer = get_answer
    
        return render_template('upload_ok.html',robot_status="待機中" ,input_answer=nor_ans,predict_answer=predict_answer,val1=time.time())
    brewflag=0
    return render_template('upload.html')

@app.route('/start_brew', methods=['GET'])  # 添加路由
def startbrew():
    global brewflag
   # if brewflag==1:
    start_Robot(1)
   # brewflag=0
    return render_template('upload_ok.html',robot_status="開始沖煮，等待沖泡完成",coffeeWeight=coffee_weight,val1=time.time())
 

if __name__ == '__main__':
    # app.debug = True
   #app.run(host='10.42.0.1', port=8080, debug=True)
   app.run(host='192.168.0.114', port=8080, debug=True)
   #app.run(host='127.0.0.1', port=8080, debug=True)
