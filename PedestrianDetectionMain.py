#encoding: utf-8
from flask import Flask
from flask import request
import base64
import json
import io
from PIL import Image

########  Func Import #########
from PedestrianDetectionFunc.PedestrianDetectionClass import *

#### PedestrianDetection Net ######
Pedestrianclass = Pedestrian()
Pedestrianclass.restore()
print("PedestrianDetection Net Ready")

app = Flask(__name__)

@app.route('/PedestrianDetection', methods=['POST'])
def PedestrianDetection_Process():
    ### Get Post Content
    ImageB64 = request.form['ImageData']
    ###
    ImageStr = base64.b64decode(ImageB64)
    mImage = io.BytesIO(ImageStr)
    mImage = Image.open(mImage)

    ResFlag, ResStr = GetPedestrianRes(mImage, Pedestrianclass)
    if ResFlag == 1:
        Res_json = json.dumps(ResStr)
        return Res_json
    else:
        Res = {'result': 'error', 'msg': 'System is busy,try later', 'errorcode': '10004'}
        return json.dumps(Res)

if __name__ == "__main__":
    app.run(host='localhost', port=8086, debug=False)