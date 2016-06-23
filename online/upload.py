import time
from flask import Flask, request, render_template
from captcha import init_model, predict

app = Flask(__name__)
model = init_model()
@app.route('/', methods=['GET', 'POST'])
def index():
	global model
	if request.method == 'POST':
		img = request.files['file']
		# img.save('data/1.jpg')
		res = predict(model, img)
		return res
	if request.method == 'GET':
		return render_template('index.html')

from werkzeug.contrib.fixers import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == '__main__':
	app.run(debug=0, host='0.0.0.0', port=5002)
