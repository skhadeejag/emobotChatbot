from flask import Flask, jsonify, request,render_template
from TestResultModel import TestResult
from MeditationRecommendations import recommended_meditation
from Chatbotfile import getresponse
import json


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
#response = ''

@app.route('/res', methods = ['POST'])
def testResult():
    #abc = 'Welcome to Emobot', result
    #global response
    requestData = request.data
    requestData = json.loads(requestData.decode('utf-8'))
    answers = requestData['answers']
    return jsonify(result = TestResult(answers))

@app.route('/meditation', methods = ['POST'])
def meditationrecommendation():
    requestData = request.data
    print(requestData)
    requestData = json.loads(requestData.decode('utf-8'))
    recommend = requestData['recommend']
    health_issue=requestData['health_issue']
    
    return jsonify(result = recommended_meditation(health_issue,recommend))

@app.route('/chat', methods = ['POST'])
def chatResponse():
    requestData = request.data
    print("chat ")
    requestData = json.loads(requestData.decode('utf-8'))
    userMessage = requestData['message']
    return jsonify(responseUser = getresponse(userMessage))

@app.route('/home', methods = ['GET'])
def home1():
    return jsonify({"message" : "Welcome to EMOBOT"})

if __name__ == '__main__':
    app.run(debug = False)
    
    

#[1,3,3,2,1,3,2]
