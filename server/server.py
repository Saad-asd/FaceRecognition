from flask import Flask, request, jsonify
import util

app = Flask(__name__)

# @app.route("/")
# def index():
#     return "Hello, World!"


# Flask route that handles HTTP GET and POST requests sent to the URL endpoint '/classify_image'
@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    # route retrieves the image data from the 'image_data' field of the request form and passes it to the 'util.classify_image' function
    image_data = request.form['image_data']

    # result of this function is returned as a JSON response
    response = jsonify(util.classify_image(image_data))

    # The response has an added header to allow cross-origin resource
    # sharing (CORS) from any origin.
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response
# CORS is a security feature implemented by web browsers to prevent
# a web page from making requests to a different domain than the one
# that served the web page. The header 'Access-Control-Allow-Origin: *'
# allows any origin to access the resource (the response in this case).
# This header is added to the response to indicate that the server is
# allowing CORS from any domain. It allows the client-side JavaScript code
# from a web page hosted on one domain to access the API hosted on another domain.
if __name__ == "__main__":
    print("Starting Python Flask Server")
    util.load_saved_artifacts()
    app.run(debug=True, port=5000)