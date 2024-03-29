from flask import Flask, request, send_file
from model import load_model, generate
from io import BytesIO

app = Flask(__name__)
model = load_model()

def serve_img(pilImage):
  imageIO = BytesIO()
  pilImage.save(imageIO, 'JPEG', quality=70)
  imageIO.seek(0)
  return imageIO


@app.route("/",methods=['GET'])
def index():
  image = generate(model)
  g = serve_img(image)
  return send_file(g, mimetype="image/jpeg")