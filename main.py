from flask import Flask, request, Response
from flask_cors import CORS, cross_origin
from urllib.parse import quote, unquote
import resourses_backend.bayes_classifier as res


import xml.etree.cElementTree as ET

app = Flask(__name__)
CORS(app)

mapping = {
    0:	'_red_heart_',
    1:	'_smiling_face_with_hearteyes_',
    2:	'_face_with_tears_of_joy_',
    3:	'_two_hearts_',
    4:	'_fire_',
    5:	'_smiling_face_with_smiling_eyes_',
    6:	'_smiling_face_with_sunglasses_',
    7:	'_sparkles_',
    8:	'_blue_heart_',
    9:	'_face_blowing_a_kiss_',
    10:	'_camera_',
    11:	'_United_States_',
    12:	'_sun_',
    13:	'_purple_heart_',
    14:	'_winking_face_',
    15:	'_hundred_points_',
    16:	'_beaming_face_with_smiling_eyes_',
    17:	'_Christmas_tree_',
    18:	'_camera_with_flash_',
    19:	'_winking_face_with_tongue_'
}


@app.route('/api/emoji', methods=['GET', 'POST', 'OPTIONS'])
def process_request():
    tweet = ''
    if 'tweet' in request.form:
        tweet = request.form['tweet']
    elif 'tweet' in request.args:
        tweet = unquote(request.args['tweet'])
    else:
        return Response('<error>tweet required</error>', 400, headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
        }, mimetype='text/xml')

    lemmas = res.normalize_tweet(tweet)
    emojis = res.classify_tweet(lemmas)
    root = ET.Element('root')
    ET.SubElement(root, 'tweet').text = tweet
    ordering = ET.SubElement(root, 'ordering')
    for i in range(0, len(emojis)):
        ET.SubElement(ordering, 'emoji',
                      position=str(i + 1),
                      label=str(emojis[i]),
                      emoji_name=mapping[emojis[i]])

    return Response(ET.tostring(root), status=200, headers={
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
    }, mimetype='text/xml')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050, debug=True)
