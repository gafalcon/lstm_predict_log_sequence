# from db import db
from datetime import datetime
import mongoengine as me

class Action(me.EmbeddedDocument):
    url = me.StringField(required=True, unique=False)
    norm_url = me.StringField(required=True, unique=False)
    method = me.StringField(required=True, unique=False)
    timestamp = me.DateTimeField(required=True, unique=False, default=datetime.utcnow)
    objType = me.StringField(required=False, unique=False)

class Seq(me.Document):
    user = me.StringField(required=True, unique=False)
    date = me.DateTimeField(required=True, unique=False, default=datetime.utcnow)
    actions = me.ListField(me.EmbeddedDocumentField(Action))
    predictions = me.ListField(me.StringField())
    prediction_probs = me.ListField(me.FloatField())

