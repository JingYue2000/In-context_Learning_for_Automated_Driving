from baseClass import Lane, Vehicle
from typing import List, Dict
from datetime import datetime
import sqlite3
import json
import os


class Scenario:
    def __init__(self, vehicleCount: int, database: str = None) -> None:
        self.lanes: Dict[str, Lane] = {}
        self.getRoadgraph()
        self.vehicles: Dict[str, Vehicle] = {}
        self.vehicleCount = vehicleCount
        self.initVehicles()

        if database:
            self.database = database
        else:
            self.database = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.db'

        if os.path.exists(self.database):
            os.remove(self.database)

        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS vehINFO(
                frame INT,
                id TEXT,
                x REAL,
                y REAL,
                lane_id TEXT,
                speedx REAL,
                speedy REAL,
                PRIMARY KEY (frame, id));"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS decisionINFO(
                frame INT PRIMARY KEY,
                scenario TEXT,
                thoughtsAndActions TEXT,
                finalAnswer TEXT,
                outputParser TEXT);"""
        )
        conn.commit()
        conn.close()

        self.frame = 0

    def getRoadgraph(self):
        for i in range(3):
            lid = 'lane_' + str(i)
            leftLanes = []
            rightLanes = []
            for j in range(i+1, 3):
                rightLanes.append('lane_' + str(j))
            for k in range(0, i):
                leftLanes.append('lane_' + str(k))
            self.lanes[lid] = Lane(
                id=lid, laneIdx=i,
                left_lanes=leftLanes,
                right_lanes=rightLanes
            )

    def initVehicles(self):
        for i in range(self.vehicleCount):
            if i == 0:
                vid = 'ego'
            else:
                vid = 'veh' + str(i)
            self.vehicles[vid] = Vehicle(id=vid)

    def updateVehicles(self, observation, frame):
        self.frame = frame
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        for i, vehicle_obs in enumerate(observation):
            presence = vehicle_obs[0]  # Assuming the first element of each vehicle observation is 'presence'
            x, y, vx, vy = vehicle_obs[1:]

            if presence == 1:  # Check if the vehicle is present
                vid = f'veh{i}' if i != 0 else 'ego'
                veh = self.vehicles[vid]
                veh.updateProperty(x, y, vx, vy)
                cur.execute(
                    '''INSERT INTO vehINFO VALUES (?,?,?,?,?,?,?);''',
                    (self.frame, vid, x, y, veh.lane_id, vx, vy)
                )
            else:
                if i != 0:  # Skip 'ego' when it's not present
                    vid = f'veh{i}'
                    self.vehicles[vid].clear()

        conn.commit()
        conn.close()

    def export2json(self):
        scenario = {}
        scenario['lanes'] = []
        scenario['vehicles'] = []
        for lv in self.lanes.values():
            scenario['lanes'].append(lv.export2json())
        scenario['ego_info'] = self.vehicles['ego'].export2json()

        for vv in self.vehicles.values():
            if vv.presence:
                scenario['vehicles'].append(vv.export2json())

        return json.dumps(scenario)

    def clear_vehicle_info(self):
        """Clears the vehINFO table in the database for a new episode."""
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute("DELETE FROM vehINFO")
        conn.commit()
        conn.close()