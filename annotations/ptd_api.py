#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from collections import defaultdict
from utils import *
import pdb

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), __file__)))
DESCS_JSON  = os.path.join(ROOT_DIR, 'data', '%s_descriptions.json')
PEOPLE_JSON = os.path.join(ROOT_DIR, 'data', '%s_people.json')
SHOTS_JSON  = os.path.join(ROOT_DIR, 'data', '%s_shots.json')
if os.path.exists(os.path.join(ROOT_DIR, 'data', 'video2fps.json')):
    VID2FPS_JSON = os.path.join(ROOT_DIR, 'data', 'video2fps.json')
else:
    VID2FPS_JSON = os.path.join(ROOT_DIR, 'data', 'video2fps.json.example')
DS = ['train', 'val', 'test']
AN_JSON = os.path.join(ROOT_DIR, 'activity_net.v1-3.min.json')

class ActivityNet(object):
    _data = None
    _video_ids = None
    def __init__(self):
        if not os.path.exists(AN_JSON):
            raise Exception('Download the annotation file of ActivityNet v1.3, and place it to "%s".' % AN_JSON)
        self._data = jsonload(AN_JSON)
        assert self._data['version'] == 'VERSION 1.3'
        self._video_ids = sorted(self._data['database'].keys())

    @property
    def video_ids(self):
        return self._video_ids

    @property
    def data(self):
        return self._data

class PTD(object):
    _descriptions = None
    _shots = None
    _people = None
    _person2desc = None
    _desc2person = None
    _person2shot = None
    _shot2person = None
    _id2shot = None
    _id2person = None
    _id2desc = None
    _vid2fps = None
    _an_data = None

    def __init__(self, dataset):
        assert dataset in DS
        self._dataset = dataset

    def description(self, index):
        return Description(self.id2desc[index], self)

    def person(self, index):
        return Person(self.id2person[index], self)

    def shot(self, index):
        return Shot(self.id2shot[index], self)

    @property
    def descriptions(self):
        if self._descriptions is None:
            self._descriptions = jsonload(DESCS_JSON % self._dataset)
        return self._descriptions

    @property
    def id2desc(self):
        if self._id2desc is None:
            self._id2desc = {}
            for desc in self.descriptions:
                self._id2desc[desc['id']] = desc
        return self._id2desc

    @property
    def people(self):
        if self._people is None:
            self._people = jsonload(PEOPLE_JSON % self._dataset)
        return self._people

    @property
    def id2person(self):
        if self._id2person is None:
            self._id2person = {}
            for person in self.people:
                self._id2person[person['id']] = person
        return self._id2person

    @property
    def shots(self):
        if self._shots is None:
            self._shots = jsonload(SHOTS_JSON % self._dataset)
        return self._shots

    @property
    def id2shot(self):
        if self._id2shot is None:
            self._id2shot = {}
            for shot in self.shots:
                self._id2shot[shot['id']] = shot
        return self._id2shot

    ### id2id ###
    @property
    def person2desc(self):
        if self._person2desc is None:
            self._person2desc = {}
            for person in self.people:
                self._person2desc[person['id']] = person['descriptions']
        return self._person2desc

    @property
    def desc2person(self):
        if self._desc2person is None:
            self._desc2person = {}
            for person in self.people:
                for desc_id in person['descriptions']:
                    self._desc2person[desc_id] = person['id']
        return self._desc2person

    @property
    def person2shot(self):
        if self._person2shot is None:
            self._person2shot = {}
            for person in self.people:
                self._person2shot[person['id']] = person['shot_id']
        return self._person2shot

    @property
    def shot2person(self):
        if self._shot2person is None:
            self._shot2person = defaultdict(list)
            for person in self.people:
                self._shot2person[person['shot_id']].append(person['id'])
        return self._shot2person

    ### others
    @property
    def vid2fps(self):
        if self._vid2fps is None:
            self._vid2fps = jsonload(VID2FPS_JSON)
        return self._vid2fps

    @property
    def an_data(self):
        if self._an_data is None:
            self._an_data = ActivityNet()
        return self._an_data

class BaseInstance(object):
    def __init__(self, info, parent):
        self._info = info
        self._parent = parent

    def __getitem__(self, index):
        return self._info[index]

    @property
    def id(self):
        return self._info['id']

class Description(BaseInstance):
    @property
    def description(self):
        return self._info['description']

    @property
    def person(self):
        person_id = self._parent.desc2person[self._info['id']]
        return self._parent.person(person_id)

    @property
    def shot(self):
        return self.person.shot

class Shot(BaseInstance):
    @property
    def video_id(self):
        return self._parent.an_data.video_ids[self._info['an_video_id']]

    @property
    def first_second(self):
        return min(self.annotated_seconds)

    @property
    def last_second(self):
        return max(self.annotated_seconds)

    @property
    def annotated_seconds(self):
        return self._info['annotated_seconds']

    @property
    def first_frame(self):
        return self.sec2frame(self.first_second)

    @property
    def last_frame(self):
        if self.video_id == 'll91M5topgU':
            return 161
        if self.video_id == 'uOmCwWVJnLQ':
            return 2999
        return self.sec2frame(self.last_second)

    @property
    def annotated_frames(self):
        return [self.sec2frame(sec) for sec in self._info['annotated_seconds']]

    @property
    def fully_annotated(self):
        return self._info['fully_annotated']

    @property
    def people(self):
        person_ids = self._parent.shot2person[self._info['id']]
        return [self._parent.person(person_id) for person_id in person_ids]

    @property
    def descriptions(self):
        descs = []
        for person in self.people:
            descs += person.descriptions
        return descs

    @property
    def fps(self):
        return self._parent.vid2fps[self.video_id]

    def sec2frame(self, sec):
        return int(round(sec * self.fps))

class Person(BaseInstance):
    @property
    def boxes(self):
        return self._info['boxes']

    @property
    def descriptions(self):
        return [self._parent.description(i) for i in self._info['descriptions']]

    @property
    def shot(self):
        shot_id = self._parent.person2shot[self._info['id']]
        return self._parent.shot(shot_id)

def demo():
    ptd = PTD('test')
    print 'Showing information of the shot of which ID is 1...'
    shot = ptd.shot(1)
    print '[SHOT] ID: %d' % shot.id
    print '[SHOT] VIDEO URL: %s' % shot.video_id
    print '[SHOT] START TIME: %s' % shot.first_second
    print '[SHOT] END TIME: %s' % shot.last_second
    print

    print 'Showing information of the person of which ID is 1...'
    person = ptd.person(1)
    print '[PERSON] ID: %d' % person.id
    print '[PERSON] PARENT SHOT ID: %s' % person.shot.id
    print '[PERSON] DESCRIPTIONS: %s' % ', '.join(['"%s"' % d.description for d in person.descriptions])
    #print

    print 'Showing information of the description of which ID is 1...'
    description = ptd.description(1)
    print '[DESCRIPTION] ID: %d' % description.id
    print '[DESCRIPTION] PARENT PERSON ID: %d' % description.person.id
    print '[DESCRIPTION] DESCRIPTION: %s' % description.description

if __name__ == '__main__':
    demo()
