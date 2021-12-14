from flask import Flask, jsonify, request,render_template

from Chatbotfile import getresponse
import json

app = Flask(__name__)
@app.route('/chat', methods = ['POST'])
def chatResponse():
    requestData = request.data
    print("chat ")
    requestData = json.loads(requestData.decode('utf-8'))
    userMessage = requestData['message']
    return jsonify(responseUser = getresponse(userMessage))
if __name__ == '__main__':
    app.run(debug = False)
    
    

#[1,3,3,2,1,3,2]
