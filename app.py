from flask import Flask, render_template


#create a Flask instance
app = Flask(__name__)

#create a route decorator
@app.route('/') 
def index():
	f_name = "Daniel"
	return render_template('index.html', f_name=f_name)

#localhost:5000/user/dani
@app.route('/user/<name>')
def user(name):
	return render_template('user.html', name=name)

#invalid url
@app.errorhandler(404)
def page_not_found(e):
	return render_template('404.html'), 404


#internal server error thing
@app.errorhandler(500)
def page_not_found(e):
	return render_template('500.html'), 500