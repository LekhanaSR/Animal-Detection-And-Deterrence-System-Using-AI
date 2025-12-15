from flask import Flask, redirect, render_template,request,make_response, url_for
import mysql.connector
from mysql.connector import Error
import sys
import os
import pandas as pd
import numpy as np
from PIL import Image
from mistralai import Mistral
from PIL import Image
import json
import pyglet
import time
import json  #json request
from werkzeug.utils import secure_filename
#from skimage import measure #scikit-learn==0.23.0  scikit-image==0.14.2
#from skimage.measure import structural_similarity as ssim #old
#from sklearn.predictor import predict
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2
import glob
from mistralai import Mistral
from PIL import Image
import json
import base64
from PIL import Image, ImageOps
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_alert_email(to_email, animal, image_filename):
    from_email = "animalpro1811@gmail.com"
    from_password = "xrmbhacozkibrgxk"  # App-specific password if using Gmail
    subject = f"Wild Animal Alert: {animal.capitalize()} Detected"
    body = f"A wild {animal} has been detected in the surveillance image ({image_filename}). Immediate action is recommended."

    # Compose message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # SMTP Setup (e.g., Gmail SMTP)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, from_password)
        server.send_message(msg)
        server.quit()
        print(f"Email alert sent to {to_email} for {animal}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")


# Add Mistral AI detection here
api_key = "WTuMOibXWmpTqjvscYHSaaCOjjXCakkJ"
model = "pixtral-large-2411"
client = Mistral(api_key=api_key)
            
def encode_image(fn):
    """Convert image file to base64."""
    image = Image.open(fn).convert("RGB")  # Open the image from file path
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def index1():
    return render_template('index.html')

@app.route('/twoform')
def twoform():
    return render_template('twoform.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/forgot')
def forgot():
    return render_template('forgot.html')

@app.route('/mainpage')
def mainpage():
    return render_template('mainpage.html')



@app.route('/regdata', methods =  ['GET','POST'])
def regdata():
    connection = mysql.connector.connect(host='localhost',database='animaldb25',user='root',password='')
    uname = request.args['uname']
    email = request.args['email']
    phn = request.args['phone']
    pssword = request.args['pswd']
    addr = request.args['addr']
    dob = request.args['dob']
    print(dob)
        
    cursor = connection.cursor()
    checkquery = "select * from userdata where Email='"+email+"'"
    cursor.execute(checkquery)
    data = cursor.fetchall()
    if len(data) > 0:
        msg="User Account Already Exists"
        resp = make_response(json.dumps(msg)) 
        connection.commit() 
        connection.close()
        cursor.close()
        return resp
    else:
        sql_Query = "insert into userdata values('"+uname+"','"+email+"','"+pssword+"','"+phn+"','"+addr+"','"+dob+"')"
        print(sql_Query)
        cursor.execute(sql_Query)
        connection.commit() 
        connection.close()
        cursor.close()
        msg="User Account Created Successfully"    
        resp = make_response(json.dumps(msg))
        return resp


def mse(imageA, imageB):
    # Compute the Mean Squared Error between the two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB, title):    
    # compute the mean squared error and structural similarity index
    m = mse(imageA, imageB)
    print(f"MSE: {m}")

    s, _ = ssim(imageA, imageB, channel_axis=-1, full=True, win_size=3)
    print(f"SSIM: {s}")
    
    return s



"""LOGIN CODE """

@app.route('/logdata', methods =  ['GET','POST'])
def logdata():
    connection=mysql.connector.connect(host='localhost',database='animaldb25',user='root',password='')
    lgemail=request.args['email']
    lgpssword=request.args['password']
    print(lgemail, flush=True)
    print(lgpssword, flush=True)
    cursor = connection.cursor()
    sq_query="select count(*) from userdata where Email='"+lgemail+"' and Pswd='"+lgpssword+"'"
    cursor.execute(sq_query)
    data = cursor.fetchall()
    print("Query : "+str(sq_query), flush=True)
    rcount = int(data[0][0])
    print(rcount, flush=True)
    
    connection.commit() 
    connection.close()
    cursor.close()
    
    if rcount>0:
        msg="Success"
        resp = make_response(json.dumps(msg))
        return resp
    else:
        msg="Failure"
        resp = make_response(json.dumps(msg))
        return resp
        

@app.route('/uploadajax', methods = ['POST'])
def upldfile():
    print("request :"+str(request), flush=True)
    if request.method == 'POST':
    
        prod_mas = request.files['first_image']
        print(prod_mas)
        filename = secure_filename(prod_mas.filename)
        prod_mas.save(os.path.join("D:\\Upload\\", filename))

        #csv reader
        fn = os.path.join("D:\\Upload\\", filename)

        count = 0
        diseaselist=os.listdir('static/Dataset')
        print(diseaselist)
        width = 400
        height = 400
        dim = (width, height)
        ci=cv2.imread("D:\\Upload\\"+ filename)
        gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/Grayscale/"+filename,gray)
        gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/Grayscale/"+filename,gray)
        #cv2.imshow("org",gray)
        #cv2.waitKey()

        thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
        cv2.imwrite("static/Threshold/"+filename,thresh)
        #cv2.imshow("org",thresh)
        #cv2.waitKey()

        lower_green = np.array([34, 177, 76])
        upper_green = np.array([255, 255, 255])
        hsv_img = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
        binary = cv2.inRange(hsv_img, lower_green, upper_green)
        cv2.imwrite("static/Binary/"+filename,gray)
        #cv2.imshow("org",binary)
        #cv2.waitKey()
        
        flagger=1
        diseasename=""
        image_base64 = encode_image(fn)

# Request wild animal name
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a wildlife expert. Identify the wild animal shown in the image below.\n\n"
                            "Return ONLY the animal name in the following JSON format:\n"
                            "{\n"
                            "  \"animal\": \"<Animal Name>\"\n"
                            "}\n\n"
                            "Do not include any explanation, text, or comments. Only return the JSON."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ],
                "response_format": {"type": "json_object"}
            }
        ]

        chat_response = client.chat.complete(
            model=model,
            messages=messages
        )

        extracted_text = chat_response.choices[0].message.content
        cleaned_data = json.loads(extracted_text[extracted_text.find('{'):extracted_text.rfind('}')+1])
        animal = cleaned_data["animal"].lower()
        # Replace this with dynamic user email from session or DB if needed
        user_email = "lekhs1803@gmail.com"
        send_alert_email(user_email, animal, filename)


        # Animal sound mapping
        animal_sounds = {
            "lion": "sounds/Elephant.mp3",
            "elephant": "sounds/Firecrackers.mp3",
            "tiger": "sounds/Elephant.mp3",
            "leopard": "sounds/Tigergrowl.mp3",
            "wolf": "sounds/Lionroaring.mp3",
            "bear": "sounds/Airhorn.mp3",
            "deer": "sounds/DogBarking.mp3",
            "fox": "sounds/DogBarking.mp3",
            "rhinoceros": "sounds/Gunshoot.mp3",
            "owl": "sounds/Eagle.mp3",
            "hippopotamus": "sounds/Elephant.mp3",
            "gorilla": "sounds/Leopardgrowl.mp3",
            "antelope": "sounds/leopard.mp3",
            "badger": "sounds/DogBarking.mp3",
            "bison": "sounds/WolfHowl.mp3",
            "coyote": "sounds/Airhorn.mp3",
            "hedgehog": "sounds/Badgergrunt.mp3",
            "hyena": "sounds/Lionroaring.mp3",
            "raccoon": "sounds/Owlsscreeching.mp3",
            "bat": "sounds/Ultrasonic.mp3",
            # Add more as needed
        }

        # Play sound using pyglet
        if animal in animal_sounds:
            sound = pyglet.media.load(animal_sounds[animal], streaming=False)
            sound.play()
            time.sleep(sound.duration)  # Keep the script alive until sound finishes
        else:
            print(f"No sound file found for '{animal}'.")

        '''
        oresized = cv2.resize(ci, dim, interpolation = cv2.INTER_AREA)
        for i in range(len(diseaselist)):
            if flagger==1:
                files = glob.glob('static/Dataset/'+diseaselist[i]+'/*')
                #print(len(files))
                for file in files:
                    # resize image
                    print(file)
                    oi=cv2.imread(file)
                    resized = cv2.resize(oi, dim, interpolation = cv2.INTER_AREA)
                    #original = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow("comp",oresized)
                    #cv2.waitKey()
                    #cv2.imshow("org",resized)
                    #cv2.waitKey()
                    #ssim_score = structural_similarity(oresized, resized, multichannel=True)
                    #print(ssim_score)
                    ssimscore=compare_images(oresized, resized, "Comparison")
                    if ssimscore>0.8:
                        diseasename=diseaselist[i]
                        flagger=0
                        break
        accuracy=predict()
        connection = mysql.connector.connect(host='localhost',database='animaldb25',user='root',password='')
        cursor = connection.cursor()
        sql_Query = "select AltMedicine,Disease from altmed where TabletName like '%"+diseasename+"%'"
        print(sql_Query)
        cursor.execute(sql_Query)
        results = cursor.fetchall()
        if(len(results)==0):
            sql_Query = "select TabletName,Disease from altmed where AltMedicine like '%"+diseasename+"%'"
            print(sql_Query)
            cursor.execute(sql_Query)
            results = cursor.fetchall()
            print(results)
        cursor.close()
        connection.commit() 
        connection.close()
        altmed=results[0][0]
        usage=results[0][1]
        
        '''
        usage=''
        altmed=''
        diseasename=""
        msg=animal+"|"+filename
        resp = make_response(json.dumps(msg))
        return resp
        

        #return render_template('mainpage.html',dname=diseasename,filename=filename,recyclables=recyclables,organics=organics )

   



@app.route('/logout')
def logout(): 
    return redirect(url_for('index'))

  
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
